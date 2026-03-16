import argparse
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = [
    'Admiration',
    'Amusement',
    'Determination',
    'Empathic Pain',
    'Excitement',
    'Joy',
]


def clip_file(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)
    cols = [col for col in SCORE_COLUMNS if col in df.columns]
    if not cols:
        raise ValueError(f'No score columns found in {input_path}')

    before_neg = int((df[cols] < 0).sum().sum())
    before_gt1 = int((df[cols] > 1).sum().sum())

    clipped = df.copy()
    for col in cols:
        clipped[col] = clipped[col].clip(lower=0.0, upper=1.0)

    after_neg = int((clipped[cols] < 0).sum().sum())
    after_gt1 = int((clipped[cols] > 1).sum().sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_csv(output_path, index=False)

    return {
        'input': str(input_path),
        'output': str(output_path),
        'before_neg': before_neg,
        'before_gt1': before_gt1,
        'after_neg': after_neg,
        'after_gt1': after_gt1,
        'rows': len(clipped),
    }


def main():
    parser = argparse.ArgumentParser(description='Clip all submission CSV score columns to [0,1].')
    parser.add_argument('--dir', default='submissions', help='Directory containing submission CSV files')
    parser.add_argument('--suffix', default='_clip01', help='Suffix appended before .csv')
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise FileNotFoundError(f'Directory not found: {root}')

    csv_files = sorted(path for path in root.glob('*.csv') if not path.stem.endswith(args.suffix))
    if not csv_files:
        print('No submission CSV files found to process.')
        return

    for csv_file in csv_files:
        output_file = csv_file.with_name(f'{csv_file.stem}{args.suffix}.csv')
        stats = clip_file(csv_file, output_file)
        print(
            f"{stats['input']} -> {stats['output']} | rows={stats['rows']} | "
            f"neg: {stats['before_neg']}->{stats['after_neg']} | "
            f">1: {stats['before_gt1']}->{stats['after_gt1']}"
        )


if __name__ == '__main__':
    main()
