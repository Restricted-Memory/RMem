import importlib
import sys
import os
import numpy as np
import datetime as datetime
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

from pathlib import Path

from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
AOT_DIR = SCRIPT_PATH.parents[1]
RMEM_DIR = SCRIPT_PATH.parents[2]

sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, str(AOT_DIR))

from . import vot

import dataloaders.video_transforms as tr
from utils.utils import Tee

import time

from utils.image import flip_tensor, save_mask, _palette
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine


def get_vot_workspace():
    workspace = os.environ.get("RMEM_VOT_WORKSPACE")
    if not workspace:
        raise RuntimeError("RMEM_VOT_WORKSPACE is not set")
    return Path(workspace).expanduser().resolve()


VOT_WORKSPACE = get_vot_workspace()
log_file = VOT_WORKSPACE / "eval_vot.log"
sys.stdout = Tee(str(log_file))
# with open(log_file, "w") as f:
#     pass

class Evaluator(object):
    def __init__(self, cfg, rank=0, vot_handle=None):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg

        print("Exp {}:".format(cfg.EXP_NAME))

        print("Use GPU {} for evaluating.".format(self.gpu))
        torch.cuda.set_device(self.gpu)

        print('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.process_pretrained_model()
        # self.result_root = os.path.join(cfg.DIR_EVALUATION,
        #                                     cfg.TEST_DATASET, 'debug1')
        method_name = "RMem"
        self.method_name = method_name
        self.vot_workspace = VOT_WORKSPACE
        self.result_root = self.vot_workspace / "result_masks" / self.method_name
        self.result_root.mkdir(parents=True, exist_ok=True)

        self.vot_handle = vot_handle

    def process_pretrained_model(self):
        cfg = self.cfg
        self.ckpt = 'unknown'
        self.model, removed_dict = load_network(self.model,
                                                cfg.TEST_CKPT_PATH,
                                                self.gpu)
        if len(removed_dict) > 0:
            print(
                'Remove {} from pretrained model.'.format(removed_dict))
        print('Load checkpoint from {}'.format(
            cfg.TEST_CKPT_PATH))

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        with torch.no_grad():
            torch.cuda.empty_cache()

            seq_total_time = 0
            seq_total_frame = 0
            seq_pred_masks = {'dense': [], 'sparse': []}
            seq_timers = []
            
            # num_frames = len(seq_dataset)
            # max_gap = int(round(num_frames / 30))
            # gap = max(max_gap, 5)
            gap = 6

            engine = build_engine(cfg.MODEL_ENGINE,
                phase='eval',
                aot_model=self.model,
                gpu_id=self.gpu,
                long_term_mem_gap=self.cfg.TEST_LONG_TERM_MEM_GAP)
            engine.eval()
            engine.long_term_mem_gap = gap
            print(f"{engine.long_term_mem_gap = }")

            frame_idx = 0
            obj_idx = None
            seq_name = None
            while True:
                imagefile = self.vot_handle.frame()
                # print(f"{imagefile = }")
                if not imagefile:
                    break

                imgname = [imagefile.split('/')[-1]]

                current_img = cv2.imread(imagefile)
                current_img = np.array(current_img, dtype=np.float32)
                current_img = current_img[:, :, [2, 1, 0]]
                height, width, _ = current_img.shape
                sample = {'current_img': current_img}

                if frame_idx == 0:
                    seq_path = Path(imagefile).parents[1]
                    print(f"{seq_path = }")
                    print(f"{self.vot_workspace = }")
                    seq_frame_num = len(os.listdir(seq_path / "color"))
                    print(f"{seq_frame_num = }")
                    seq_name = str(seq_path.stem)
                    print(f"{seq_name = }")
                    engine.long_term_mem_gap = int(round(seq_frame_num / 30))
                    engine.long_term_mem_gap = max(engine.long_term_mem_gap, 5)
                    print(f"{engine.long_term_mem_gap = }")
                    # if seq_name == '8234_grind_beans':
                    #     engine.long_term_mem_gap = 10
                    # else:
                    #     engine.long_term_mem_gap = gap

                    objects = self.vot_handle.objects()
                    # print(f"{objects.shape = }")
                    obj_nums = [len(objects)]
                    obj_idx = list(range(len(objects) + 1))
                    print(f"{obj_idx = }")

                    print(
                        f'\nGPU {self.gpu} - Processing Seq {seq_name}')

                    squeezed_label = np.zeros((height, width))
                    annotation_path = self.vot_workspace / "annotations" / seq_name
                    annotation_path.mkdir(parents=True, exist_ok=True)
                    # print(f"{objects[0].dtype = }")
                    # print(f"{objects[0].shape = }")
                    # print(f"{np.unique(objects[0]) = }")
                    for idx in range(len(obj_idx)):
                        obj_id = obj_idx[idx]
                        if obj_id == 0:
                            continue
                        mask = objects[idx - 1]
                        Image.fromarray(mask * 255).save(annotation_path / f"{idx}.png")
                        y_pad = height - mask.shape[0]
                        x_pad = width - mask.shape[1]
                        mask = np.pad(mask, ((0, y_pad), (0, x_pad)))
                        squeezed_label += (mask * idx).astype(np.uint8)
                    objects = squeezed_label
                    objects_image = Image.fromarray(objects).convert('P')
                    objects_image.putpalette(_palette)
                    objects_image.save(annotation_path / f"mask.png")
                    sample['current_label'] = objects
                    objects = torch.from_numpy(objects).unsqueeze(0)
                    # print(f"{objects.shape = }")

                    seq_pred_masks['dense'].append({
                        'path':
                        os.path.join(self.result_root, seq_name,
                                    imgname[0].split('.')[0] + '.png'),
                        'mask':
                        objects,
                        'obj_idx':
                        obj_idx
                    })

                sample['meta'] = {
                    'seq_name': seq_name,
                    'obj_num': obj_nums[0],
                    'current_name': imgname,
                    'height': height,
                    'width': width,
                    'flip': False,
                    'obj_idx': obj_idx
                }

                sample = eval_transforms(sample)[0]

                current_img = sample['current_img']
                current_img = current_img.cuda(self.gpu,
                                                non_blocking=True).unsqueeze(0)

                if 'current_label' in sample.keys():
                    current_label = sample['current_label'].cuda(
                        self.gpu, non_blocking=True).float().unsqueeze(0)
                else:
                    current_label = None

                #############################################################

                if frame_idx == 0:
                    _current_label = F.interpolate(
                        current_label,
                        size=current_img.size()[2:],
                        mode="nearest")
                    engine.add_reference_frame(current_img,
                                                _current_label,
                                                frame_step=0,
                                                obj_nums=obj_nums)
                else:
                    seq_timers.append([])
                    now_timer = torch.cuda.Event(
                        enable_timing=True)
                    now_timer.record()
                    seq_timers[-1].append((now_timer))

                    pred_logit = engine.match_propogate_one_frame(
                        current_img, output_size=(height, width))

                    pred_prob = torch.softmax(pred_logit, dim=1)
                    pred_label = torch.argmax(pred_prob,
                                                dim=1,
                                                keepdim=True).float()

                    current_label = F.interpolate(
                        pred_label,
                        size=engine.input_size_2d,
                        mode="nearest")
                    engine.update_memory(current_label)

                    now_timer = torch.cuda.Event(enable_timing=True)
                    now_timer.record()
                    seq_timers[-1].append((now_timer))

                    vot_preds = []
                    for obj_id in obj_idx:
                        if obj_id == 0:
                            continue
                        mask = (pred_label == obj_id).cpu().detach().squeeze().numpy()
                        obj_label = (mask * obj_id).astype(np.uint8)
                        vot_preds.append(obj_label)

                    self.vot_handle.report(vot_preds)

                    # Save result
                    seq_pred_masks['dense'].append({
                        'path':
                        os.path.join(self.result_root, seq_name,
                                    imgname[0].split('.')[0] + '.png'),
                        'mask':
                        pred_label,
                        'obj_idx':
                        obj_idx
                    })

                frame_idx += 1

            seq_dir = os.path.join(self.result_root, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            # Save result
            for mask_result in seq_pred_masks['dense'] + seq_pred_masks[
                    'sparse']:
                save_mask(mask_result['mask'].squeeze(0).squeeze(0),
                            mask_result['path'], mask_result['obj_idx'])
            del (seq_pred_masks)

            for timer in seq_timers:
                torch.cuda.synchronize()
                one_frametime = timer[0].elapsed_time(timer[1]) / 1e3
                seq_total_time += one_frametime
                seq_total_frame += 1
            del (seq_timers)

            seq_avg_time_per_frame = seq_total_time / seq_total_frame
            max_mem = torch.cuda.max_memory_allocated(
                device=self.gpu) / (1024.**3)
            print(
                "GPU {} - Seq {} - FPS: {:.2f}. Max Mem: {:.2f}G"
                .format(self.gpu, seq_name, 1. / seq_avg_time_per_frame,
                        max_mem))

    def print_log(self, string):
        if self.rank == 0:
            print(string)


# sys.stdout = Tee(f"eval_vot_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.log")

engine_config = importlib.import_module('configs.pre_vost_2')
cfg = engine_config.EngineConfig('aotplus', 'r50_deaotl')

cfg.TEST_EMA = True

cfg.TEST_GPU_ID = 0
cfg.TEST_GPU_NUM = 1

cfg.TEST_CKPT_PATH = str(
    AOT_DIR / "pretrain_models" / "aotplus_R50_DeAOTL_Temp_pe_Slot_4_ema_20000.pth"
)

cfg.TEST_DATASET = 'vost'
cfg.TEST_DATASET_SPLIT = 'val'

cfg.TEST_FLIP = False
cfg.TEST_MULTISCALE = [1.0]
# cfg.TEST_MULTISCALE = [1.0, 1.1, 1.2, 0.9, 0.8]

cfg.TEST_MIN_SIZE = None
cfg.TEST_MAX_SIZE = 1080.0

handle = vot.VOT("mask", multiobject=True)

# Initiate a evaluating manager
evaluator = Evaluator(rank=0,
                        cfg=cfg,
                        vot_handle=handle)
# Start evaluation
evaluator.evaluating()
