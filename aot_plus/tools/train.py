import sys

import importlib
import os
import random
import sys

sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

from utils.utils import Tee, copy_codes, make_log_dir

import torch.multiprocessing as mp

from networks.managers.trainer import Trainer
from get_config import get_config


def main_worker(gpu, cfg, enable_amp=True, exp_name='default', log_dir=None):
    if cfg.FIX_RANDOM:
        random_seed = 1 << gpu
        print(f"[{gpu}] : Fix random seed {random_seed}")
        import os
        os.environ['CUDNN_DETERMINISTIC'] = '1'
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
        import random
        random.seed(random_seed+1)
        import numpy as np
        np.random.seed(random_seed+2)
        import torch
        torch.manual_seed(random_seed+3)
        torch.cuda.manual_seed(random_seed+4)
        torch.cuda.manual_seed_all(random_seed+5)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    # Initiate a training manager
    if gpu == 0:
        sys.stdout = Tee(os.path.join(log_dir, "print.log"))
    trainer = Trainer(rank=gpu, cfg=cfg, enable_amp=enable_amp)
    # Start Training
    trainer.sequential_training()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VOS")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dist_url', type=str, default='')
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--pretrained_path', type=str, default='')

    parser.add_argument('--datasets', nargs='+', type=str, default=[])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1.)
    parser.add_argument('--start_step', type=int, default=-1.)

    parser.add_argument('--log', type=str, default='./logs')

    parser.add_argument('--debug_fix_random', action='store_true')
    parser.set_defaults(debug_fix_random=False)
    parser.add_argument('--fix_random', action='store_true')
    parser.set_defaults(fix_random=False)

    args = parser.parse_args()

    cfg = get_config(args.stage, args.exp_name, args.model)

    log_dir = make_log_dir(args.log, cfg.EXP_NAME)
    copy_codes(log_dir)
    # sys.stdout = Tee(os.path.join(log_dir, "print.log"))

    if len(args.datasets) > 0:
        cfg.DATASETS = args.datasets

    cfg.DIST_START_GPU = args.start_gpu
    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num
    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path != '':
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr

    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step

    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    if args.dist_url == '':
        cfg.DIST_URL = 'tcp://127.0.0.1:123' + str(random.randint(0, 9)) + str(
            random.randint(0, 9))
    else:
        cfg.DIST_URL = args.dist_url

    cfg.save_self()

    setattr(cfg, "DEBUG_FIX_RANDOM", args.debug_fix_random)
    setattr(cfg, "FIX_RANDOM", args.fix_random)

    if cfg.TRAIN_GPUS == 1:
        main_worker(0, cfg, args.amp, args.exp_name, log_dir=log_dir) 
    else:
        # Use torch.multiprocessing.spawn to launch distributed processes
        mp.spawn(main_worker, nprocs=cfg.TRAIN_GPUS, args=(cfg, args.amp, args.exp_name, log_dir))


if __name__ == '__main__':
    main()
