# if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing. set_start_method("spawn")
import os
import datetime
import time
from typing import Union
import warnings
import math

import torch
from torch.nn.modules.module import Module
import torch.utils.data
import torchvision
import torchvision.transforms
from . import utils, transforms, presets
from .sampler import RASampler
from torch import Tensor, nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from ...base import LossManager, Model, addprop
from ...procedures.adversarial import FGSMExample
# from ...scheduler.sine import SineAnnealingScheduler

import inspect

class ForwardingDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):

    def get_properties(self, obj=None):
        """Return a list of property names of an object."""
        if obj is None: obj = self.module
        return [name for name, value in inspect.getmembers(obj.__class__, predicate=inspect.isdatadescriptor) if type(value) is property]

    def create_property(self, prop_name):
        def getter(self):
            return getattr(self.module, prop_name)

        def setter(self, value):
            setattr(self.module, prop_name, value)

        return property(getter, setter)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except:
            if name == 'iteration':
                print("getattr", name, self.module.__class__)
            return getattr(self.module, name)
    def __setattr__(self, name: str, value) -> None:
        try: 
            super().__setattr__(name, value)
        except:
            if name == 'iteration':
                print("setattr", name, self.module.__class__)
            setattr(self.module, name, value)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, model_without_ddp=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    if hasattr(model, 'losses'): model.losses.reset()

    save_every_iteration = 125 if epoch < 5 else 300
    if epoch < 1:
        save_every_iteration = 25

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image: torch.Tensor; target: torch.Tensor
        assert args.batch_size_per_proc % args.physical_batch_size == 0
        chunks = int(math.ceil(args.batch_size_per_proc // args.physical_batch_size))
        output = []; loss = []
        optimizer.zero_grad()
        for minibatch_image, minibatch_target in zip(image.chunk(chunks, dim=0), target.chunk(chunks, dim=0)):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                with torch.autograd.profiler.record_function("forward propagation"):
                    minibatch_output = model(minibatch_image)
                    assert minibatch_output.requires_grad
                    if len(minibatch_target.shape) > 1:
                        minibatch_target = minibatch_target.argmax(dim=-1)
                    if args.augmented_flatness_only and args.correct_samples_only:
                        minibatch_loss = nn.functional.cross_entropy(minibatch_output, minibatch_target, reduce=None, reduction='none', label_smoothing=args.label_smoothing) / chunks
                        minibatch_correct = minibatch_output.argmax(dim=-1) == minibatch_target
                        minibatch_loss = (minibatch_loss * minibatch_correct).mean()
                        model.losses.observe(minibatch_correct.float().mean(), 'augmented_flatness_acc')
                    else:
                        minibatch_loss = criterion(minibatch_output, minibatch_target) / chunks

                assert minibatch_loss.requires_grad
                with torch.autograd.profiler.record_function("backward"):
                    if scaler is not None:
                        scaler.scale(minibatch_loss).backward()
                    else:
                        minibatch_loss.backward()
                if hasattr(model, 'after_minibatch_backward'):
                    model.after_minibatch_backward()
            output.append(minibatch_output);loss.append(minibatch_loss)
        
        with torch.no_grad():
            output = torch.cat(output, dim=0); loss = torch.stack(loss, dim=0).sum()


        if scaler is not None:
            scaler.unscale_(optimizer)
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            if hasattr(model, 'after_backward'): model.after_backward()

            scaler.step(optimizer)
            scaler.update()
        else:
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            if hasattr(model, 'after_backward'): model.after_backward()

            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        if (model.iteration + 1) % save_every_iteration == 0 and args.fine_grained_checkpoints and epoch < 10:
            print("saving")
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "epoch": epoch,
                "args": args,
            } # params only to save storage
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'save', f"param_{epoch}_{model.iteration + 1}.pth"))


        if hasattr(model, 'losses'):
            model.losses.observe(loss, 'loss')
            model.losses.observe(acc1 / 100, 'acc', 1)
            model.losses.observe(acc5 / 100, 'acc', 5)
            model.losses.observe(optimizer.param_groups[0]["lr"], 'lr')
            if model.iteration % args.log_per_step == 0:
                model.losses.log_losses(model.iteration)
            model.iteration += 1
            if not args.augmented_flatness_only:
                model.losses.reset()
            if args.max_iteration is not None and model.iteration >= args.max_iteration:
                return



def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", adversarial=False, n_step=None, adversarial_eps=0.0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    if hasattr(model, 'losses'): model.losses.reset()

    if adversarial:
        def adv_test(pred, Y):
            return pred
        adv = FGSMExample(None, model, torch.nn.CrossEntropyLoss(), adversarial_eps, adv_test, tqdm=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    num_processed_samples = 0
    turn_off_grad = torch.inference_mode if not adversarial else torch.no_grad
    with turn_off_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
                
            if adversarial:
                with torch.enable_grad():
                    _, adv_image, _, output, adv_output = list(adv.run(len(target), generator=True, data=[(image, target)]))[0]
            else:
                output = model(image)
            
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            if adversarial:
                acc1, acc5 = utils.accuracy(adv_output, target, topk=(1, 5))
                metric_logger.meters["advs_acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["advs_acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
            if hasattr(model, 'after_testing_step'):
                model.after_testing_step()
            if n_step is not None and num_processed_samples / batch_size > n_step:
                break
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        False and hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    if hasattr(model, 'losses'):
        if len(log_suffix) > 0:
            suffix = '_' + log_suffix
        else:
            suffix=''
        model.losses.observe(metric_logger.acc1.global_avg / 100, 'test_acc' + suffix, 1)
        model.losses.observe(metric_logger.acc5.global_avg / 100, 'test_acc' + suffix, 5)
        if adversarial:
            model.losses.observe(metric_logger.advs_acc1.global_avg / 100, 'test_advs_acc' + suffix, 1)
            model.losses.observe(metric_logger.advs_acc5.global_avg / 100, 'test_advs_acc' + suffix, 5)
        model.losses.log_losses(model.iteration, testing=True)
        model.losses.reset()
        model.after_testing()
        model.epoch += 1

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    if adversarial:
        print(f"Adversarial {header} Acc@1 {metric_logger.advs_acc1.global_avg:.3f} Acc@5 {metric_logger.advs_acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    def cifar10_target_transform(y: Tensor):
        return y.argmax(dim=-1)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        if 'cifar10' in traindir:
            dataset = torchvision.datasets.CIFAR10(
                traindir,
                True,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                    backend=args.backend,
                ),
            )
        else:
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                    backend=args.backend,
                ),
            )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
            )
        if 'cifar10' in valdir:
            dataset_test = torchvision.datasets.CIFAR10(
                valdir,
                False,
                preprocessing,
            )
        else:
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


from .mymodel import get_imagenet1k_model


def main(args):
    if args.physical_batch_size is None:
        args.physical_batch_size = args.batch_size_per_proc
    if args.physical_epochs is None:
        args.physical_epochs = args.epochs
    if args.from_scratch or args.finetune:
        args.start_epoch = 0
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = args.gpu if args.distributed else "cuda"

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if 'cifar10' not in args.data_path:
        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")
    else:
        train_dir = args.data_path
        val_dir = args.data_path
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_proc,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.physical_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.workers > 0 else False
    )

    data_loader_test_trainig_samples = torch.utils.data.DataLoader(
        dataset, batch_size=args.physical_batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.workers > 0 else False
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint["epoch"] + 1 if args.resume else args.start_epoch
    print(f"Starting from epoch {args.start_epoch}")

    print("Creating model")
    if args.model in ['vanilla', 'sparsified']:
        model = get_imagenet1k_model(args.model, data_loader, args, start_epoch=checkpoint['epoch'] + 1 if args.resume else 0, epoch_size=len(data_loader), max_epoch_mixing_activations=args.activation_mixing_epoch)
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)


    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    # custom_keys_weight_decay.append(("biases", 0)) # no weight decay for the first one of dual biases so that it moves freely in flat minima
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    if not args.post_training_only:
        parameters = utils.set_weight_decay(
            model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
            initial_lr=args.lr,
        )
    else:
        model.requires_grad_(False)
        for name, p in model.named_parameters():
            if 'pos_embedding' in name:
                p.requires_grad = True
        def get_epoch(s: str):
            s = s.split('/')[-1]
            s = s[:s.find('.pth')]
            if s.find('param_') >= 0:
                s = s[s.find('_') + len('_'):]
                s = s[:s.find('_')]
                return  int(s)
            else:
                s = s[s.find('model_') + len('model_'):]
                if s.isdigit():
                    return int(s) + 1
                else:
                    return 299 + 1
        def get_iteration(s: str):
            s = s.split('/')[-1]
            if 'param' in s:
                return int(s.split('_')[-1].split('.')[0])
            else:
                return get_epoch(s) * len(data_loader)
        model.iteration = get_iteration(args.finetune or args.resume)
        parameters = [{"params": model.parameters(), "initial_lr": 0, "momentum": 0, "weight_decay": 0.0}]
        args.max_iteration = (args.max_iteration or 1000000000) + model.iteration 
        model.epoch = get_epoch(args.finetune or args.resume)

    opt_name = args.opt.lower()

    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop, AdamW and Adam are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'sineannealinglr':
        main_lr_scheduler = SineAnnealingScheduler(
            optimizer,
            T_max=args.epochs - args.lr_warmup_epochs,
            last_epoch=args.start_epoch - 1,
            warmup_phase=args.warmup_phase,
            verbose=True
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - args.lr_warmup_epochs, 
            eta_min=args.lr_min, 
            last_epoch=args.start_epoch - 1,
            verbose=True
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = addprop(ForwardingDistributedDataParallel(model, device_ids=[args.gpu]))
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size_per_proc * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only and not args.post_training_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # lr_scheduler.step(lr_scheduler.last_epoch)
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler and not args.post_training_only:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.post_training_only:
        for g in optimizer.param_groups:
            g['initial_lr'] = 0.0
            g['lr'] = 0.0
        lr_scheduler.base_lrs = [0] * len(lr_scheduler.base_lrs)

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    def train_epochs():
        print("Start training")
        if args.finetune and not args.resume and not args.no_testing:
            if args.test_training_samples > 0:
                evaluate(model, criterion, data_loader_test_trainig_samples, device=device, adversarial=args.adversarial_testing, log_suffix="Training Samples", n_step=args.test_training_samples, adversarial_eps=args.adversarial_eps)
            evaluate(model, criterion, data_loader_test, device=device, adversarial=args.adversarial_testing, adversarial_eps=args.adversarial_eps)
        model.train()
        assert any([p.requires_grad for p in iter(model.parameters())])
        with torch.autograd.profiler.profile(use_cuda=True, with_flops=True, with_stack=True, enabled=False) as prof:
            start_time = time.time()
            for epoch in range(args.start_epoch, args.physical_epochs):
                if args.augmented_flatness_only:
                    initial_step = model.iteration
                with torch.autograd.profiler.record_function("training"):
                    if args.distributed:
                        train_sampler.set_epoch(epoch)
                    train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, model_without_ddp)
                    if args.augmented_flatness_only:
                        model.losses.log_losses(global_step=initial_step, csv=True)
                        model.losses.writer.flush()
                        break
                    if hasattr(model, 'iteration') and args.max_iteration is not None and model.iteration >= args.max_iteration:
                        break
                    lr_scheduler.step()
                    if not args.no_testing:
                        if args.test_training_samples > 0:
                            evaluate(model, criterion, data_loader_test_trainig_samples, device=device, adversarial=args.adversarial_testing, log_suffix="Training Samples", n_step=args.test_training_samples, adversarial_eps=args.adversarial_eps)
                        evaluate(model, criterion, data_loader_test, device=device, adversarial=args.adversarial_testing, adversarial_eps=args.adversarial_eps)
                    if model_ema:
                        evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
                    if args.output_dir:
                        checkpoint = {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        }
                        if model_ema:
                            checkpoint["model_ema"] = model_ema.state_dict()
                        if scaler:
                            checkpoint["scaler"] = scaler.state_dict()
                        if epoch % args.save_every_epoch == 0:
                            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'save', f"model_{epoch}.pth"))
                        utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'save', "checkpoint.pth"))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
        if prof is not None:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    if args.compile:
        train_epochs = torch.compile(train_epochs, mode='max-autotune')
    train_epochs()

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--title", type=str, default="ImageNet1K")

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="vanilla", type=str, help="model name")
    parser.add_argument(
        "-b", "--batch-size-per-proc", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--physical-batch-size", default=None, type=int, help="size of mini-batches (default: None, same value as --batch-size)")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="(logical) number of total epochs to run")
    parser.add_argument("--physical-epochs", default=None, type=int, help="actual number of total epochs to run (default: None, same value as --epochs)")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default=None, type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--log_per_step", type=int, default=10, help="the number of steps between two logging")
    parser.add_argument("--save-every-epoch", type=int, default=5, help="the number of epochs between two saving")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--zeroth-bias-clipping", type=float, default=0.1, help="the upperbound of absolute values of entries in implicit adversarial sample layer.")
    parser.add_argument("--wide", action="store_true", help="turn on wide MLP for transformers")
    parser.add_argument("--dont-resume-lr-schedulers", action="store_true")
    parser.add_argument("--warmup_phase", type=float, default=1.0, help="length relative to 2 Pi, indicating how many epochs are under warmup")
    parser.add_argument("--max-iteration", type=int, default=None, help="maximum number of iterations, only used in profiling")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--restricted-affine", action='store_true', help="whether to force off bias and force scaling factors >=1 in LayerNorm layers.")
    parser.add_argument("--magic-synapse", action='store_true')
    parser.add_argument("--magic-synapse-rho", type=float, default='0.1')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--activation-mixing-epoch", type=int, default=10)
    parser.add_argument("--mixed-activation", action="store_true")
    parser.add_argument("--layernorm-uplifting-epoch", type=int, default=10)
    parser.add_argument("--adversarial-testing", action="store_true")
    def parse_epsilon(eps: str):
        if isinstance(eps, str):
            return eval(eps)
        return eps
    parser.add_argument("--adversarial-eps", action="store_const", const=parse_epsilon, dest="adversarial_eps", default="1/255")
    parser.add_argument("--test-training-samples", type=int, default=0, help="step number to test training samples in every evaluation")
    parser.add_argument("--magic-residual", action="store_true")
    parser.add_argument("--post-training-only", action='store_true')
    parser.add_argument("--gradient-density-only", action="store_true")
    parser.add_argument("--augmented-flatness-only", action="store_true")
    parser.add_argument("--correct-samples-only", action="store_true")
    parser.add_argument("--no-testing", action="store_true")
    parser.add_argument("--fine-grained-checkpoints", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
