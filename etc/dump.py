import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from argparse import ArgumentParser

def get_subtag(acc: EventAccumulator):
    str_tag:str = acc.path.split('/')[-1]
    try:
        return int(str_tag)
    except:
        return str(str_tag)

def tabulate_events(dpath, filter=None):
    def _filter(dname):
        if filter is None or len(filter) == 0:
            return True
        try:
            return dname in filter
        except:
            return filter(dname)

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, dname)) and _filter(dname)]
    summary_iterators = sorted(summary_iterators, key=lambda acc: get_subtag(acc))

    tags = summary_iterators[0].Tags()['scalars']
    print('tags:', tags)
    print('scalars:', len(summary_iterators))

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags, (it.Tags()['scalars'], tags)

    out = {}

    for tag in tags:
        subtags = [get_subtag(acc) for acc in summary_iterators]
        records = []
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            step = events[0].step
            records.append({'step': step, **{subtag: e.value for subtag, e in zip(subtags, events)}})
        out[tag] = pd.DataFrame.from_records(records)
    return out


parser = ArgumentParser()
parser.add_argument('--source-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--filter-dname', type=str, nargs='+')

args = parser.parse_args()

steps = tabulate_events(args.source_dir, filter=args.filter_dname)

# Save each tag to a separate CSV
os.makedirs(args.output_dir, exist_ok=True)
for tag, df in steps.items():
    df.to_csv(os.path.join(args.output_dir, f"{tag}.csv"))
