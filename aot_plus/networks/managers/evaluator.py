import copy
import gc
import os
import time
import datetime as datetime
import json
from typing import List
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
torch.set_printoptions(linewidth=328)
from tqdm import tqdm

from dataloaders.eval_datasets import YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST, VOST_Test, LONG_VIDEOS_Test
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine
from networks.engines.aot_engine import AOTEngine, AOTInferEngine


class Evaluator(object):
    def __init__(
        self,
        cfg,
        rank=0,
        seq_queue=None,
        info_queue=None,
    ):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        self.print_log(f"Exp {cfg.EXP_NAME}:")
        self.print_log(
            json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print(f"Use GPU {self.gpu} for evaluating.")
        torch.cuda.set_device(self.gpu)

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg

        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return

        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(
                        map(
                            lambda x: int(x.split('_')[-1].split('.')[0]),
                            ckpts,
                        ))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log(f'No checkpoint in {cfg.DIR_CKPT}.')
                    exit()
            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(
                cfg.DIR_CKPT,
                f'save_step_{ckpt}.pth',
            )
            self.model, removed_dict = load_network(
                self.model,
                cfg.TEST_CKPT_PATH,
                self.gpu,
            )
            if len(removed_dict) > 0:
                self.print_log(
                    f'Remove {removed_dict} from pretrained model.')
            self.print_log(
                f'Load latest checkpoint from {cfg.TEST_CKPT_PATH}')
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(
                self.model,
                cfg.TEST_CKPT_PATH,
                self.gpu,
            )
            if len(removed_dict) > 0:
                self.print_log(
                    f'Remove {removed_dict} from pretrained model.')
            self.print_log(
                f'Load checkpoint from {cfg.TEST_CKPT_PATH}')

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(
                cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                cfg.MODEL_ALIGN_CORNERS,
            ),
            tr.MultiToTensor()
        ])

        if cfg.TEST_DATASET_SPLIT == "test":
            cfg.DIR_EVALUATION = cfg.DIR_TEST

        eval_name = f'{cfg.TEST_DATASET}_{cfg.TEST_DATASET_SPLIT}_{cfg.EXP_NAME}_{cfg.STAGE_NAME}_ckpt_{self.ckpt}'

        if cfg.TEST_EMA:
            eval_name += '_ema'
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms_' + str(cfg.TEST_MULTISCALE).replace(
                '.', 'dot').replace('[', '').replace(']', '').replace(
                    ', ', '_')

        if 'youtubevos' in cfg.TEST_DATASET:
            year = int(cfg.TEST_DATASET[-4:])
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION,
                cfg.TEST_DATASET, eval_name,
                'Annotations',
            )
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest

                self.result_root_sparse = os.path.join(
                    cfg.DIR_EVALUATION,
                    cfg.TEST_DATASET,
                    eval_name + '_sparse',
                    'Annotations',
                )
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    f'{eval_name}_sparse.zip')
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.dataset = youtubevos_test(
                root=cfg.DIR_YTB,
                year=year,
                split=split,
                transform=eval_transforms,
                result_root=self.result_root,
            )

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION,
                cfg.TEST_DATASET, eval_name,
                'Annotations', resolution,
            )
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION,
                cfg.TEST_DATASET, eval_name,
                'Annotations', resolution,
            )
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root,
            )

        elif cfg.TEST_DATASET == 'long_videos':
            eval_name = cfg.EVAL_NAME
            resolution = '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,
                                            eval_name)
            self.dataset = LONG_VIDEOS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_LONG_VIDEOS,
                year=2017,
                transform=eval_transforms,
                full_resolution=False,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'vost':
            eval_name = cfg.EVAL_NAME
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION,
                cfg.TEST_DATASET, eval_name,
            )
            self.dataset = VOST_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_VOST,
                transform=eval_transforms,
                result_root=self.result_root,
                is_oracle=True if hasattr(
                    cfg, "ORACLE") and cfg.ORACLE else False,
            )
            # self.dataset.seqs = self.dataset.seqs[ : 8]
            # self.dataset.seqs = self.dataset.seqs[self.dataset.seqs.index('4030_cut_broccoli') : ]
            # self.dataset.seqs = os.listdir("./results/aotplus_R50_AOTL_No_mem_gap/pre_vost/eval/vost/test")
            # self.dataset.seqs = self.dataset.seqs[self.dataset.seqs.index('1210_cut_garlic') : ]
            print(f"{self.dataset.seqs = }")
        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION,
                cfg.TEST_DATASET, eval_name,
                'Annotations',
            )
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log(
            f'Eval {cfg.EXP_NAME} on {cfg.TEST_DATASET} {cfg.TEST_DATASET_SPLIT}:')
        self.source_folder = os.path.join(
            cfg.DIR_EVALUATION, cfg.TEST_DATASET,
            eval_name, 'Annotations',
        )
        self.zip_dir = os.path.join(
            cfg.DIR_EVALUATION, cfg.TEST_DATASET,
            f'{eval_name}.zip',
        )
        print(f"{self.result_root = }")
        if not os.path.exists(self.result_root):
            try:
                print(f"making dir {self.result_root = }")
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log(f'Failed to mask dir: {self.result_root}.')
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0
        processed_video_num = 0
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        start_eval_time = time.time()

        if self.seq_queue is not None:
            if self.rank == 0:
                for seq_idx in range(total_video_num):
                    self.seq_queue.put(seq_idx)
                for _ in range(self.gpu_num):
                    self.seq_queue.put('END')
            coming_seq_idx = self.seq_queue.get()

        all_engines: List[AOTInferEngine] = []
        with torch.no_grad():
            for seq_idx, seq_dataset in enumerate(self.dataset):
                video_num += 1

                if self.seq_queue is not None:
                    if coming_seq_idx == 'END':
                        break
                    elif coming_seq_idx != seq_idx:
                        continue
                    else:
                        coming_seq_idx = self.seq_queue.get()

                processed_video_num += 1

                for engine in all_engines:
                    engine.restart_engine()

                seq_name = seq_dataset.seq_name
                print(
                    f'\nGPU {self.gpu} - Processing Seq {seq_name} [{video_num}/{total_video_num}]:')
                gc.collect()
                torch.cuda.empty_cache()

                seq_dataloader = DataLoader(
                    seq_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=cfg.TEST_WORKERS,
                    pin_memory=True,
                )

                if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                    images_sparse = seq_dataset.images_sparse
                    seq_dir_sparse = os.path.join(
                        self.result_root_sparse,
                        seq_name,
                    )
                    if not os.path.exists(seq_dir_sparse):
                        os.makedirs(seq_dir_sparse)

                seq_total_time = 0
                seq_total_frame = 0
                seq_pred_masks = {'dense': [], 'sparse': []}
                seq_timers = []

                num_frames = len(seq_dataset)
                max_gap = int(round(num_frames / 30))
                gap = max(max_gap, 5)
                if cfg.NO_MEMORY_GAP:
                    gap = int(round(gap / 4))
                print(f"{num_frames = }  {gap = }  long term memry frames : {num_frames / gap}")

                for frame_idx, samples in enumerate(tqdm(seq_dataloader)):

                    all_preds = []
                    new_obj_label = None

                    for aug_idx in range(len(samples)):
                        if len(all_engines) <= aug_idx:
                            all_engines.append(
                                build_engine(
                                    cfg.MODEL_ENGINE,
                                    phase='eval',
                                    aot_model=self.model if aug_idx==0 else copy.deepcopy(self.model),
                                    gpu_id=self.gpu,
                                    long_term_mem_gap=self.cfg.
                                    TEST_LONG_TERM_MEM_GAP,
                                ))
                            all_engines[-1].eval()

                        engine = all_engines[aug_idx]
                        engine.long_term_mem_gap = gap

                        sample = samples[aug_idx]

                        is_flipped = sample['meta']['flip']

                        obj_nums = sample['meta']['obj_num']
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        obj_idx = sample['meta']['obj_idx']

                        obj_nums = [int(obj_num) for obj_num in obj_nums]
                        obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                        current_img = sample['current_img']
                        current_img = current_img.cuda(
                            self.gpu,
                            non_blocking=True,
                        )
                        sample['current_img'] = current_img

                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(
                                self.gpu, non_blocking=True).float()
                        else:
                            current_label = None

                        #############################################################

                        if frame_idx == 0:
                            _current_label = F.interpolate(
                                current_label,
                                size=current_img.size()[2:],
                                mode="nearest").int()
                            engine.add_reference_frame(
                                current_img,
                                _current_label,
                                frame_step=0,
                                obj_nums=obj_nums,
                            )
                            pred_prob = _current_label
                        else:
                            if aug_idx == 0:
                                seq_timers.append([])
                                now_timer = torch.cuda.Event(
                                    enable_timing=True)
                                now_timer.record()
                                seq_timers[-1].append((now_timer))

                            if self.cfg.USE_MASK:
                                if self.cfg.PREV_PROBE:
                                    pred_logit = engine.match_propogate_one_frame(
                                        current_img, mask=pred_prob, output_size=(ori_height, ori_width))
                                elif self.cfg.ORACLE:
                                    _current_label = F.interpolate(
                                        current_label,
                                        size=current_img.size()[2:],
                                        mode="nearest",
                                    ).int()
                                    pred_logit = engine.match_propogate_one_frame(
                                        current_img, mask=_current_label, output_size=(ori_height, ori_width))
                                    current_label = None
                                else:
                                    raise Exception("Unexpeted !")
                            else:
                                pred_logit = engine.match_propogate_one_frame(
                                    current_img, output_size=(ori_height, ori_width))
                            if cfg.DEBUG_FIX_RANDOM:
                                print(f"\n [{self.rank}] : {frame_idx = } {pred_logit[0, :7, 100, 100] = }")

                            if is_flipped:
                                pred_logit = flip_tensor(pred_logit, 3)

                            pred_prob = torch.softmax(pred_logit, dim=1)
                            all_preds.append(pred_prob)

                            if not is_flipped and current_label is not None and new_obj_label is None:
                                new_obj_label = current_label

                    if frame_idx > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        pred_prob = torch.mean(all_preds, dim=0, keepdim=True)
                        pred_label = torch.argmax(pred_prob,
                                                  dim=1,
                                                  keepdim=True).float()

                        # LSTT = self.model.LSTT
                        # if hasattr(LSTT.layers[0], "record_T"):
                        #     h, w = engine.aot_engines[0].enc_size_2d
                        #     inner_pred_logits = engine.aot_engines[0].pred_id_logits
                        #     inner_pred_logits = F.interpolate(inner_pred_logits, size=(h, w), mode="bilinear").detach().cpu()
                        #     inner_pred_label = torch.argmax(inner_pred_logits, dim=1).squeeze()
                        #     print(f"{LSTT.layers[0].record_T = }")
                        #     print(f"{engine.aot_engines[0].long_memories_indexes = }")
                        #     saved_layer_memory = {
                        #             "h": h,
                        #             "w": w,
                        #             "ori_height": ori_height,
                        #             "ori_width": ori_width,
                        #             "inner_pred_label": inner_pred_label.detach().cpu(),
                        #             "long_mem_len": LSTT.layers[0].record_T,
                        #             "memory_indices": engine.aot_engines[0].long_memories_indexes,
                        #             "attn_weights": [],
                        #             "short_attn_weights": [],
                        #     }
                        #     output_seq_folder = os.path.join(self.result_root, seq_name)
                        #     if LSTT.layers[0].record_T > 1:
                        #         for lstt_layer in LSTT.layers:
                        #             attn_values = np.reshape(lstt_layer.attn_values, (h, w, -1))
                        #             attn_indices_T, attn_indices_hw = lstt_layer.attn_indices
                        #             attn_indices_h, attn_indices_w = np.unravel_index(attn_indices_hw, (h, w))
                        #             attn_indices = np.stack([attn_indices_T, attn_indices_h, attn_indices_w], axis=-1)
                        #             attn_indices = np.reshape(attn_indices, (h, w, -1, 3))
                        #             saved_layer_memory["attn_weights"].append({
                        #                 "attn_values": attn_values,
                        #                 "attn_indices": attn_indices,
                        #             })
                        #             attn_values = np.reshape(lstt_layer.short_attn_values, (h, w, -1))
                        #             attn_indices_h, attn_indices_w = np.unravel_index(lstt_layer.short_attn_indices, (h, w))
                        #             attn_indices = np.stack([attn_indices_h, attn_indices_w], axis=-1)
                        #             attn_indices = np.reshape(attn_indices, (h, w, -1, 2))
                        #             saved_layer_memory["short_attn_weights"].append({
                        #                 "attn_values": attn_values,
                        #                 "attn_indices": attn_indices,
                        #             })
                        #         torch.save(saved_layer_memory, os.path.join(output_seq_folder, f"{pathlib.Path(imgname[0]).stem}_layer_mem.pt"))

                        if new_obj_label is not None:
                            keep = (new_obj_label == 0).float()
                            pred_label = pred_label * \
                                keep + new_obj_label * (1 - keep)
                            new_obj_nums = [int(pred_label.max().item())]

                            if cfg.TEST_FLIP:
                                flip_pred_label = flip_tensor(pred_label, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_img = samples[aug_idx]['current_img']

                                current_label = flip_pred_label if samples[
                                    aug_idx]['meta']['flip'] else pred_label
                                current_label = F.interpolate(
                                    current_label,
                                    size=engine.input_size_2d,
                                    mode="nearest")
                                engine.add_reference_frame(
                                    current_img,
                                    current_label,
                                    obj_nums=new_obj_nums,
                                    frame_step=frame_idx,
                                )
                        else:
                            if cfg.TEST_FLIP:
                                flip_pred_label = flip_tensor(
                                    pred_label, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_label = flip_pred_label if samples[
                                    aug_idx]['meta']['flip'] else pred_label
                                current_label = F.interpolate(
                                    current_label,
                                    size=engine.input_size_2d,
                                    mode="nearest",
                                )
                                engine.update_memory(current_label)

                        now_timer = torch.cuda.Event(enable_timing=True)
                        now_timer.record()
                        seq_timers[-1].append((now_timer))

                        if cfg.TEST_FRAME_LOG:
                            torch.cuda.synchronize()
                            one_frametime = seq_timers[-1][0].elapsed_time(
                                seq_timers[-1][1]) / 1e3
                            obj_num = obj_nums[0]
                            print(
                                f"GPU {self.gpu} - Frame: {imgname[0].split('.')[0]} - Obj Num: {obj_num}, Time: {int(one_frametime * 1e3)}ms")

                        # Save result
                        seq_pred_masks['dense'].append({
                            'path':
                            os.path.join(
                                self.result_root, seq_name,
                                imgname[0].split('.')[0] + '.png',
                            ),
                            'mask':
                            pred_label.detach().cpu(),
                            'obj_idx':
                            obj_idx
                        })
                        if 'all_frames' in cfg.TEST_DATASET_SPLIT and imgname in images_sparse:
                            seq_pred_masks['sparse'].append({
                                'path':
                                os.path.join(
                                    self.result_root_sparse, seq_name,
                                    imgname[0].split('.')[0] + '.png',
                                ),
                                'mask':
                                pred_label.detach().cpu(),
                                'obj_idx':
                                obj_idx
                            })

                # Save result
                for mask_result in seq_pred_masks['dense'] + seq_pred_masks[
                        'sparse']:
                    save_mask(
                        mask_result['mask'].squeeze(0).squeeze(0),
                        mask_result['path'], mask_result['obj_idx'],
                    )
                del (seq_pred_masks)

                for timer in seq_timers:
                    torch.cuda.synchronize()
                    one_frametime = timer[0].elapsed_time(timer[1]) / 1e3
                    seq_total_time += one_frametime
                    seq_total_frame += 1
                del (seq_timers)

                seq_avg_time_per_frame = seq_total_time / seq_total_frame
                total_time += seq_total_time
                total_frame += seq_total_frame
                total_avg_time_per_frame = total_time / total_frame
                total_sfps += seq_avg_time_per_frame
                avg_sfps = total_sfps / processed_video_num
                max_mem = torch.cuda.max_memory_allocated(
                    device=self.gpu) / (1024.**3)
                print(
                    f"GPU {self.gpu} - Seq {seq_name} - FPS: {1. / seq_avg_time_per_frame:.2f}. All-Frame FPS: {1. / total_avg_time_per_frame:.2f}, All-Seq FPS: {1. / avg_sfps:.2f}, Max Mem: {max_mem:.2f}G")

        if self.seq_queue is not None:
            if self.rank != 0:
                self.info_queue.put({
                    'total_time': total_time,
                    'total_frame': total_frame,
                    'total_sfps': total_sfps,
                    'processed_video_num': processed_video_num,
                    'max_mem': max_mem,
                })
            print('Finished the evaluation on GPU {}.'.format(self.gpu))
            if self.rank == 0:
                for _ in range(self.gpu_num - 1):
                    info_dict = self.info_queue.get()
                    total_time += info_dict['total_time']
                    total_frame += info_dict['total_frame']
                    total_sfps += info_dict['total_sfps']
                    processed_video_num += info_dict['processed_video_num']
                    max_mem = max(max_mem, info_dict['max_mem'])
                all_reduced_total_avg_time_per_frame = total_time / total_frame
                all_reduced_avg_sfps = total_sfps / processed_video_num
                print(
                    "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                    .format(list(range(self.gpu_num)),
                            1. / all_reduced_total_avg_time_per_frame,
                            1. / all_reduced_avg_sfps, max_mem))
        else:
            print(
                f"GPU {self.gpu} - All-Frame FPS: {1. / total_avg_time_per_frame:.2f}, All-Seq FPS: {1. / avg_sfps:.2f}, Max Mem: {max_mem:.2f}G")

        if self.rank == 0:
            # zip_folder(self.source_folder, self.zip_dir)
            # self.print_log('Saving result to {}.'.format(self.zip_dir))
            # if 'all_frames' in cfg.TEST_DATASET_SPLIT:
            #     zip_folder(self.result_root_sparse, self.zip_dir_sparse)
            end_eval_time = time.time()
            total_eval_time = str(
                datetime.timedelta(
                    seconds=int(end_eval_time - start_eval_time)))
            self.print_log(f"Total evaluation time: {total_eval_time}")

    def print_log(self, string):
        if self.rank == 0:
            print(string)
