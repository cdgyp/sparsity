"""
I used tensorboardX for experiment tracking, but during my experiments after resuming the code version changes and the tag naming of tensorboard  writer is also different, resulting in some scalars split under two tags. I want you to write codes in order to merge these groups of scalars. Write a function to process multiple groups scalars.  Each resulting scalar after splitting looks like:
the_record_directory
├── activation_concentration_test/
    ├── [obs]activation
    │   ├── 0
    │   │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    │   ├── 1
    │   │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    │   ├── ...
    │   └── 9
    │       └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    └── [obs]pre_activation
        ├── 0
        │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
        ├── 1
        │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
        ├── ...
        └── 9
            └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
├── activation_concentration_(test)/
    ├── [obs]activation
    │   ├── 0
    │   │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    │   ├── 1
    │   │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    │   ├── ...
    │   └── 9
    │       └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
    └── [obs]pre_activation
        ├── 0
        │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
        ├── 1
        │   └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
        ├── ...
        └── 9
            └── events.out.tfevents.1695004082.autodl-container-eca8478bc6-d02ac33a
There are multiple *groups* of resulting scalars so I expect the input of the merging codes to be multiple groups of paths pointing to scalars directories like `activation_concentration_test` above. One group should be a list of paths exactly pointing to things like `activation_concentration_test` because these scalars are mixed with non-split ones of the same level as `activation_concentration_test` that require no merging.
Summarize my requirements and list your plan before start coding!
I dont have tensorflow or independent installation of tensorboard. Use tensorboardX only!

no, the same tag under different `path` in one group should be merged correspondingly. For example, in one group, the path1/[obs]activations/1 should be merged with path2/[obs]activations/1 instead of path1/[obs]activations/2!
And dont delete the unmerged ones
Also write a function for each group and wrap it to have tidier codes
"""


from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import fnmatch

def get_all_subdirs(root_dir):
    """
    Recursively get all subdirectories starting from root_dir.
    Only those subdirectories that contain a "*.tfevents.*" file are considered.
    """
    subdirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fnmatch.fnmatch(filename, "*.tfevents.*") for filename in filenames):
            relative_dir = os.path.relpath(dirpath, root_dir)
            subdirs.append(relative_dir)
    return subdirs

def form_correspondence(group):
    """
    Given a group, form a correspondence between scalars that should be merged.
    """
    all_subdirs = [(root, get_all_subdirs(root)) for root in group]

    correspondences = {}

    for root, subdirs in all_subdirs:
        for subdir in subdirs:
            if subdir not in correspondences:
                correspondences[subdir] = []
            correspondences[subdir].append(os.path.join(root, subdir))
    
    return list(correspondences.values())  # Only returning grouped full paths

def merge_scalars(paths, output_dir):
    """
    Given multiple paths in a correspondence, merge scalars from each path and write to the output directory.
    Each path is assumed to have only one tag, but these tags may be different across paths.
    The tag from the first path is used for the merged scalars.
    """
    accumulators = [EventAccumulator(path).Reload() for path in paths]

    writer = SummaryWriter(os.path.join(output_dir, os.path.relpath(paths[0], paths[0].split(os.sep)[0])))  # Using the relative structure from the first path
    
    merged_scalars = []

    for acc in accumulators:
        tags = acc.Tags()['scalars']
        # Assert that there's only one tag in each subdir
        assert len(tags) == 1, f"Expected only one tag in {acc.path}, but found {len(tags)}."

        tag = tags[0]
        merged_scalars.extend(acc.Scalars(tag))

    # Sorting by step to ensure scalars are in order
    merged_scalars.sort(key=lambda x: x.step)
    
    # Using the tag name from the first path
    output_tag = accumulators[0].Tags()['scalars'][0]

    for scalar in merged_scalars:
        writer.add_scalar(output_tag, scalar.value, scalar.step)
    
    writer.close()


def process_and_merge(groups, output_dir):
    """
    For each group of paths, find the correspondence and then merge the scalars.
    """
    for group in groups:
        correspondences = form_correspondence(group)
        for correspondence in correspondences:
            merge_scalars(correspondence, output_dir)


groups = []
for stage in ["train", "test"]:
    for sub_dir in ["activation", "pre_activation"]:
        groups.append(
            [
                f"runs/imagenet1k/from_scratch5/sparsified/20230915-133443/_activation_concentration_({stage})/[obs]{sub_dir}",
                f"runs/imagenet1k/from_scratch5/sparsified/20230915-133443/_activation_concentration_{stage}/[obs]{sub_dir}"
            ]
        )

process_and_merge(groups, 'runs/merged')