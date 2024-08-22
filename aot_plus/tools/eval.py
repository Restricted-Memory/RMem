import sys

import importlib
import sys
import os
from pathlib import Path

sys.path.append('.')
sys.path.append('..')

from utils.utils import Tee, copy_codes, make_log_dir

import torch
import torch.multiprocessing as mp

from networks.managers.evaluator import Evaluator
from get_config import get_config


def main_worker(gpu, cfg, seq_queue=None, info_queue=None, enable_amp=False):
    if cfg.FIX_RANDOM:
        random_seed = 1
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
    # Initiate a evaluating manager
    evaluator = Evaluator(rank=gpu,
                          cfg=cfg,
                          seq_queue=seq_queue,
                          info_queue=info_queue)
    # Start evaluation
    if enable_amp:
        with torch.cuda.amp.autocast(enabled=True):
            evaluator.evaluating()
    else:
        evaluator.evaluating()


#python tools/eval.py --exp_name aotplus --stage pre_vost --model r50_aotl --dataset vost --split val --gpu_num 8 --ckpt_path pretrain_models/aotplus.pth --ms 1.0 1.1 1.2 0.9 0.8
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval VOS")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    parser.add_argument('--result_path', type=str, required=True)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--split', type=str, default='')

    parser.add_argument('--no_ema', action='store_true')
    parser.set_defaults(no_ema=False)

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--log', type=str, default='./eval_logs')
    parser.add_argument('--eval_name', type=str, default='debug')

    parser.add_argument('--debug_fix_random', action='store_true')
    parser.set_defaults(debug_fix_random=False)
    parser.add_argument('--fix_random', action='store_true')
    parser.set_defaults(fix_random=False)

    parser.add_argument('--former_mem_len', type=int, default=1)
    parser.add_argument('--latter_mem_len', type=int, default=9999)

    args = parser.parse_args()


    sys.path.append(f"{args.result_path}/")
    if Path(f"{args.result_path}/config.py").is_file():
        import config
        cfg = config.Config()
    else:
        cfg = get_config(args.stage, args.exp_name, args.model)


    log_dir = make_log_dir(args.log, cfg.EXP_NAME)
    sys.stdout = Tee(os.path.join(log_dir, "print.log"))

    cfg.TEST_EMA = not args.no_ema

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = args.gpu_num

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset

    if args.split != '':
        cfg.TEST_DATASET_SPLIT = args.split

    cfg.TEST_FLIP = args.flip
    cfg.TEST_MULTISCALE = args.ms

    cfg.TEST_MIN_SIZE = None
    cfg.TEST_MAX_SIZE = args.max_resolution * 800. / 480.

    setattr(cfg, "EVAL_NAME", args.eval_name)
    setattr(cfg, "DEBUG_FIX_RANDOM", args.debug_fix_random)
    setattr(cfg, "FIX_RANDOM", args.fix_random)

    cfg.FORMER_MEM_LEN = args.former_mem_len
    cfg.LATTER_MEM_LEN = args.latter_mem_len

    if args.gpu_num > 1:
        mp.set_start_method('spawn')
        seq_queue = mp.Queue()
        info_queue = mp.Queue()
        mp.spawn(main_worker,
                 nprocs=cfg.TEST_GPU_NUM,
                 args=(cfg, seq_queue, info_queue, args.amp))
    else:
        main_worker(0, cfg, enable_amp=args.amp)


if __name__ == '__main__':
    main()
