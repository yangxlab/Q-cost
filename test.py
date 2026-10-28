import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.models.q_cost.functional import configure_runner
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a Q-cost DINO checkpoint')
    parser.add_argument('config', help='MMDetection configuration with the desired test dataloader and evaluator')
    parser.add_argument('checkpoint')
    parser.add_argument('--params', default=str(ROOT / 'mmdet/models/q_cost_config.py'))
    parser.add_argument('--work-dir')
    parser.add_argument('--out')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir')
    parser.add_argument('--wait-time', type=float, default=2)
    parser.add_argument('--param-options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if args.out and not args.out.endswith(('.pkl', '.pickle')):
        parser.error('--out must end in .pkl or .pickle')
    return args


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    parameters = Config.fromfile(args.params)
    if args.param_options:
        parameters.merge_from_dict(args.param_options)
    cfg = configure_runner(Config.fromfile(args.config), parameters)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir or cfg.get('work_dir') or str(ROOT / 'work_dirs/q_cost_test')
    cfg.load_from = args.checkpoint
    cfg.resume = False
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    runner = RUNNERS.build(cfg) if 'runner_type' in cfg else Runner.from_cfg(cfg)
    if args.out:
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))
    runner.test()


if __name__ == '__main__':
    main()
