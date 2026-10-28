import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mmengine.config import Config, DictAction
from mmengine.dist import barrier, is_main_process
from mmengine.runner import Runner

from mmdet.models.q_cost.functional import configure_runner
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train MMDetection DINO with Q-cost')
    parser.add_argument('config', help='MMDetection runner configuration with the current domain dataloaders')
    parser.add_argument('--params', default=str(ROOT / 'mmdet/models/q_cost_config.py'))
    parser.add_argument('--domain-id', type=int)
    parser.add_argument('--work-dir')
    parser.add_argument('--load-from')
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--param-options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if args.load_from and args.resume:
        parser.error('--load-from starts a new stage; --resume continues an existing stage')
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    parameters = Config.fromfile(args.params)
    if args.param_options:
        parameters.merge_from_dict(args.param_options)
    cfg = configure_runner(Config.fromfile(args.config), parameters, args.domain_id)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    domain = cfg.model.q_cost_cfg.domain_id
    cfg.work_dir = args.work_dir or cfg.get('work_dir') or str(ROOT / 'work_dirs' / f'q_cost_domain_{domain}')
    cfg.resume = False
    if args.load_from:
        cfg.load_from = args.load_from
    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else args.resume
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
    if cfg.model.q_cost_cfg.enabled:
        for hook in cfg.custom_hooks:
            if hook.type == 'QCostIncrementalHook':
                hook.domain_id = domain
        if not any(h.type == 'QCostIncrementalHook' for h in cfg.custom_hooks):
            raise ValueError('QCostIncrementalHook must be enabled to manage incremental state')
    runner = RUNNERS.build(cfg) if 'runner_type' in cfg else Runner.from_cfg(cfg)
    runner.train()
    if is_main_process():
        runner.save_checkpoint(runner.work_dir, 'q_cost_final.pth',
                               save_optimizer=True, save_param_scheduler=True)
    barrier()


if __name__ == '__main__':
    main()
