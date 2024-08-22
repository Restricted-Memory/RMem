import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict, List

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.models.aot import AOT

from networks.layers.position import get_temporal_positional_encoding

from networks.debug import debug
USE_ATTEN_WEIGHT_DROP = True
class AOTEngine(nn.Module):
    def __init__(
        self,
        aot_model: AOT,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
    ):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.restart_engine()

    def forward(
        self,
        all_frames,
        all_masks,
        batch_size,
        obj_nums,
        step=0,
        tf_board=False,
        use_prev_pred=False,
    ):  # only used for training
        if self.losses is None:
            self._init_losses()

        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(
            self.aux_step - step, 0.) / self.aux_step

        if hasattr(self.cfg, "PREV_PROBE") and self.cfg.PREV_PROBE:
            self.split_all_frames = torch.split(
                all_frames, self.batch_size, dim=0)
            self.generate_offline_masks(all_masks)
            self.total_offline_frame_num = len(self.offline_masks)
            self.add_reference_frame(
                img=self.split_all_frames[self.frame_step],
                mask=self.offline_masks[self.frame_step],
                frame_step=0,
                obj_nums=obj_nums,
            )
        else:
            self.offline_encoder(all_frames, all_masks)

            self.add_reference_frame(
                frame_step=0,
                obj_nums=obj_nums,
            )

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        self.match_propogate_one_frame(
            mask=self.offline_masks[self.frame_step])
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        for _ in range(self.total_offline_frame_num - 2):
            curr_loss = self.update_short_term_memory(
                curr_mask,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step],
                    self.offline_ignore_masks[self.frame_step],
                ),
                step=step,
            )
            if curr_loss is not None:
                curr_losses.append(curr_loss)
            self.match_propogate_one_frame(mask=curr_prob)
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)
        # mask = (torch.cat(curr_losses, dim=0) > 0).float()
        # pred_loss = torch.cat(curr_losses, dim=0).sum(-1) / (mask.sum(-1) + 0.000001)

        loss = aux_weight * aux_loss + pred_loss
        loss_print_text = f"loss {loss} = aux_weight {aux_weight} * aux_loss {aux_loss} + pred_loss {pred_loss}"
        if hasattr(self.cfg, "VAR_LOSS_WEIGHT"):
            var_loss = self.AOT.get_var_loss()
            loss = loss + self.cfg.VAR_LOSS_WEIGHT * var_loss
            loss_print_text += f" + {self.cfg.VAR_LOSS_WEIGHT} * var_loss {var_loss}  |  frame_num {self.total_offline_frame_num}"
        if step % self.cfg.TRAIN_LOG_STEP == 0:
            print(loss_print_text)

        all_pred_mask = aux_masks + curr_masks

        all_frame_loss = aux_losses + curr_losses

        boards = {'image': {}, 'scalar': {}}  # type:Dict[str,Dict[str,List]]

        return loss, all_pred_mask, all_frame_loss, boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS,
        )
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            if hasattr(self.cfg, 'USE_MASK') and self.cfg.USE_MASK:
                curr_enc_embs = self.AOT.encode_image(img, mask=mask)
            else:
                curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask, curr_ignore_mask = one_hot_mask(
                mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
            curr_ignore_mask = self.offline_ignore_masks[frame_step]
        else:
            curr_one_hot_mask = None
            curr_ignore_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size

        # extract backbone features
        if self.cfg.USE_MASK:
            if self.cfg.ORACLE:
                self.offline_enc_embs = self.split_frames(
                    self.AOT.encode_image(all_frames, all_masks), self.batch_size)
        else:
            self.offline_enc_embs = self.split_frames(
                self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            self.generate_offline_masks(all_masks)

        if self.input_size_2d is None:
            self.update_size(
                all_frames.size()[2:],
                self.offline_enc_embs[0][-1].size()[2:],
            )

    def generate_offline_masks(self, masks):
        offline_one_hot_masks, offline_ignore_masks = one_hot_mask(
            masks, self.max_obj_num)
        self.offline_masks = list(
            torch.split(masks, self.batch_size, dim=0))
        self.offline_one_hot_masks = list(
            torch.split(offline_one_hot_masks, self.batch_size, dim=0))
        self.offline_ignore_masks = list(
            torch.split(offline_ignore_masks, self.batch_size, dim=0))

    def assign_identity(self, one_hot_mask, ignore_mask=None):
        if ignore_mask is None:
            ignore_mask = torch.zeros(
                one_hot_mask.shape[0], 1, one_hot_mask.shape[2], one_hot_mask.shape[3],
                device=torch.device('cuda', self.gpu_id),
            )
        if self.cfg.MODEL_IGNORE_TOKEN:
            non_ignored = (ignore_mask == 0).float()
            one_hot_mask[:, 0, :, :] = one_hot_mask[
                :, 0, :, :] * non_ignored.squeeze()
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum(
                'bohw,bot->bthw', one_hot_mask,
                self.id_shuffle_matrix,
            )
        if self.cfg.MODEL_IGNORE_TOKEN:
            one_hot_mask = torch.cat((one_hot_mask, ignore_mask), 1)

        id_emb = self.AOT.get_id_emb(one_hot_mask).view(
            self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(
        self,
        img=None,
        mask=None,
        frame_step=-1,
        obj_nums=None,
        img_embs=None,
    ):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
                img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(
                None, mask, frame_step)
            curr_enc_embs = img_embs

        self.ref_enc_embs = curr_enc_embs
        if mask is not None:
            self.ref_mask = mask
        else:
            self.ref_mask = self.offline_masks[0]
        self.ref_mask_downsample = F.interpolate(
            self.ref_mask.float(), self.ref_enc_embs[0].shape[-2:], mode='nearest')
        self.ref_one_hot_mask_downsample, _ = one_hot_mask(
            self.ref_mask_downsample, self.max_obj_num)

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_one_hot_mask = curr_one_hot_mask

        if self.pos_emb is None:
            self.pos_emb = self.AOT.get_pos_emb(curr_enc_embs[-1]).expand(
                self.batch_size, -1, -1, -1,
            ).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)
        if self.cfg.TIME_ENCODE and (not self.cfg.TIME_ENCODE_NORM):
            self.temporal_encoding = get_temporal_positional_encoding(
                max_sequence_len=32,
                channels=curr_enc_embs[-1].size()[1],
                device=curr_enc_embs[-1].device,
                is_normalize=True,
                scale=1.57,
                # is_debug=True,
            )
        else:
            self.temporal_encoding = None

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        if self.AOT.use_temporal_pe:
            temporal_pos_emb = torch.cat((self.AOT.cur_pos_emb, self.AOT.mem_pos_emb), dim=0) 
        else:
            temporal_pos_emb = None
        curr_lstt_output = self.AOT.LSTT_forward(
            curr_enc_embs,
            curr_id_emb,
            pos_emb=self.pos_emb,
            size_2d=self.enc_size_2d,
            temporal_encoding=temporal_pos_emb,
        )

        self.last_mem_step = frame_step
        self.AOT.init_LSTT_memory(size_2d=self.enc_size_2d)
        self.long_memories_indexes.append(self.frame_step)

        self.decode_current_logits(curr_enc_embs, curr_lstt_output)

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, step=0):
        if curr_id_emb is None:
            curr_ignore_mask = None
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask, curr_ignore_mask = one_hot_mask(
                    curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(
                curr_one_hot_mask, curr_ignore_mask)

        is_update_long_memory = False
        if (not (hasattr(self.cfg, "NO_LONG_MEMORY") and self.cfg.NO_LONG_MEMORY)) \
                and self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            # print(f"update long memory {self.frame_step = }")
            is_update_long_memory = True
            self.last_mem_step = self.frame_step
        self.AOT.update_short_term_memory(
            curr_id_emb=curr_id_emb,
            short_term_mem_skip=self.short_term_mem_skip,
            size_2d=self.enc_size_2d,
            is_update_long_memory=is_update_long_memory,
        )
        if is_update_long_memory:
            self.long_memories_indexes.append(self.frame_step)
            debug(f"{self.long_memories_indexes = }")
            # if self.AOT.LSTT.long_term_memories[0][0].size(0) > \
            #     (self.cfg.FORMER_MEM_LEN + self.cfg.LATTER_MEM_LEN):
            pred_id_logits = F.interpolate(
                self.pred_id_logits,
                size=self.enc_size_2d,
                mode="bilinear",
                align_corners=True,
            )
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            foreground_proba = 1 - pred_prob[:, 0:1, ...]
            self.AOT.LSTT.restrict_long_memories(
                former_memory_len=self.cfg.FORMER_MEM_LEN,
                latter_memory_len=self.cfg.LATTER_MEM_LEN,
                use_atten_weight=USE_ATTEN_WEIGHT_DROP and (not self.training),
                long_memories_indexes=self.long_memories_indexes,
                foreground_proba=foreground_proba,
            )
            debug(f"{self.AOT.LSTT.long_term_memories[0][0].size(0) = }")
        if self.cfg.REVERSE_INFER:
            if self.frame_step == 1:
                self.first_short_memories = [
                    [mem_k_v[0].detach().clone(), mem_k_v[1].detach().clone()] for mem_k_v in self.AOT.LSTT.short_term_memories]
            if is_update_long_memory:
                long_memory_remove_1st = [
                    [mem_k_v[0][1:, ...], mem_k_v[1][1:, ...]] for mem_k_v in self.AOT.LSTT.long_term_memories
                ]
                curr_lstt_output = self.AOT.LSTT_forward(
                    curr_embs=self.ref_enc_embs,
                    curr_id_emb=None,
                    pos_emb=self.pos_emb,
                    size_2d=self.enc_size_2d,
                    is_outer_memory=True,
                    outer_long_memories=long_memory_remove_1st,
                    outer_short_memories=self.first_short_memories,
                )
                pred_id_logits = self.decode_current_logits(self.ref_enc_embs, curr_lstt_output)
                if self.training:
                    curr_loss, _ = self.generate_loss_mask(
                        self.ref_mask, step, return_prob=False)
                    curr_loss = self.cfg.REVERSE_LOSS * curr_loss
                else:
                    curr_loss = None

                return curr_loss

    def match_propogate_one_frame(self, img=None, img_embs=None, mask=None, output_size=None):
        self.frame_step += 1
        # print(f"{self.frame_step = }")
        if (not self.enable_offline_enc) and (img is None):
            img = self.split_all_frames[self.frame_step]
        if img_embs is None:
            if hasattr(self.cfg, "USE_MASK") and self.cfg.USE_MASK:
                curr_enc_embs, _ = self.encode_one_img_mask(
                    img, mask, self.frame_step)
            else:
                curr_enc_embs, _ = self.encode_one_img_mask(
                    img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs

        if self.cfg.TIME_ENCODE_NORM:
            self.temporal_encoding = get_temporal_positional_encoding(
                max_sequence_len=self.AOT.LSTT.long_term_memories[0][0].size(0)+1,
                channels=curr_enc_embs[-1].size()[1],
                device=curr_enc_embs[-1].device,
                is_normalize=True,
                scale=1.,
                # is_debug=True,
            )
        if self.AOT.use_temporal_pe:
            temporal_pos_emb = torch.cat((self.AOT.cur_pos_emb, self.AOT.mem_pos_emb), dim=0) 
        else:
            temporal_pos_emb = None
        curr_lstt_output = self.AOT.LSTT_forward(
            curr_enc_embs,
            None,
            pos_emb=self.pos_emb,
            size_2d=self.enc_size_2d,
            temporal_encoding=temporal_pos_emb,
            save_atten_weights=USE_ATTEN_WEIGHT_DROP and (not self.training),
            # save_atten_weights=True,
        )

        return self.decode_current_logits(curr_enc_embs, curr_lstt_output, output_size=output_size)

    def decode_current_logits(self, curr_enc_embs, curr_lstt_embs, output_size=None,):
        pred_id_logits = self.AOT.decode_id_logits(
            curr_lstt_embs,
            curr_enc_embs,
        )

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum(
                'bohw,bto->bthw', pred_id_logits,
                self.id_shuffle_matrix,
            )

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num+1):] = - \
                1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(
                pred_id_logits,
                size=output_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        return pred_id_logits

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(
            self.pred_id_logits,
            size=output_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits

        pred_id_logits = F.interpolate(
            pred_id_logits,
            size=gt_mask.size()[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            now_label = gt_mask[batch_idx].long()
            now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
            label_list.append(now_label.long())
            logit_list.append(now_logit)

        total_loss = 0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + loss_weight * \
                loss(logit_list, label_list, step)

        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()
            return loss, mask

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.AOT.clear_LSTT_memory()
        self.long_memories_indexes = []
        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_ignore_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class AOTInferEngine(nn.Module):
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
    ):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip

        self.aot_engines: List[AOTEngine] = []

        self.restart_engine()

    def restart_engine(self):
        for engine in self.aot_engines:
            engine.restart_engine()
        self.aot_engines = []
        self.obj_nums = None

    def separate_mask(self, mask):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask]

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(
            torch.cat(bg_logits, dim=1),
            dim=1,
            keepdim=True,
        )
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_aot_obj_num])

        bg_prob = torch.prod(
            torch.cat(bg_probs, dim=1),
            dim=1,
            keepdim=True,
        )
        merged_prob = torch.cat(
            [bg_prob] + fg_probs,
            dim=1,
        ).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOTEngine(
                self.AOT, self.gpu_id,
                self.long_term_mem_gap,
                self.short_term_mem_skip,
            )
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks = self.separate_mask(mask)
        img_embs = None
        for aot_engine, separated_mask in zip(
            self.aot_engines,
            separated_masks,
        ):
            aot_engine.add_reference_frame(
                img,
                separated_mask,
                obj_nums=[self.max_aot_obj_num],
                frame_step=frame_step,
                img_embs=img_embs,
            )

        self.update_size()

    def match_propogate_one_frame(self, img=None, mask=None, output_size=None):
        img_embs = None
        all_logits = []
        for aot_engine in self.aot_engines:
            logits = aot_engine.match_propogate_one_frame(
                img, img_embs=img_embs, mask=mask, output_size=output_size)
            all_logits.append(logits)
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask):
        separated_masks = self.separate_mask(curr_mask)
        for aot_engine, separated_mask in zip(
            self.aot_engines,
            separated_masks,
        ):
            aot_engine.update_short_term_memory(separated_mask)

    def update_size(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_size_2d = self.aot_engines[0].enc_size_2d
        self.enc_hw = self.aot_engines[0].enc_hw
