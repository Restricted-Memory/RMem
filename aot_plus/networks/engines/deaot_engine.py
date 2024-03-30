import numpy as np

from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.engines.aot_engine import AOTEngine, AOTInferEngine


class DeAOTEngine(AOTEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 layer_loss_scaling_ratio=2.):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip)
        self.layer_loss_scaling_ratio = layer_loss_scaling_ratio

class DeAOTInferEngine(AOTInferEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = DeAOTEngine(self.AOT, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
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
