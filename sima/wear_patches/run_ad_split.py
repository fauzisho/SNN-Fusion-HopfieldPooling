#!/usr/bin/env python3
"""Run Attend-and-Discriminate training for a single LOSO split (SLURM array)."""
import argparse
import os
import sys
import tempfile

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Base YAML config with all splits listed')
    parser.add_argument('--split', type=int, required=True, help='LOSO split index (1-18)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run-id', default='', type=str)
    parser.add_argument(
        '--out-dir',
        default='',
        type=str,
        help='If set, write logs/checkpoints here (sets WEAR_LOG_ROOT)',
    )
    parser.add_argument('--ckpt-freq', type=int, default=-1)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--val-freq', type=int, default=1)
    args = parser.parse_args()

    if not 1 <= args.split <= 18:
        raise ValueError(f'--split must be between 1 and 18, got {args.split}')

    with open(args.config, 'r') as handle:
        config = yaml.load(os.path.expandvars(handle.read()), Loader=yaml.FullLoader)  # ${NWEAR_DATA} in configs

    if config.get('name') != 'attendanddiscriminate':
        raise ValueError(f"Config name must be attendanddiscriminate, got {config.get('name')}")

    splits = config['anno_json']
    if args.split > len(splits):
        raise ValueError(f'Config has {len(splits)} splits, requested split {args.split}')

    config['anno_json'] = [splits[args.split - 1]]
    run_id = args.run_id or f'ad_split{args.split:02d}_seed{args.seed}'

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        os.environ['WEAR_LOG_ROOT'] = os.path.abspath(args.out_dir)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
        split_config = handle.name

    os.chdir(REPO_ROOT)
    try:
        from main import main as wear_main

        wear_main(argparse.Namespace(
            config=split_config,
            run_id=run_id,
            neptune=False,
            seed=args.seed,
            ckpt_freq=args.ckpt_freq,
            resume=args.resume,
            gpu='cuda:0',
            val_freq=args.val_freq,
        ))
    finally:
        os.unlink(split_config)


if __name__ == '__main__':
    main()
