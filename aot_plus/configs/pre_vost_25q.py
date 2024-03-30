import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'pre_vost_25q'

        self.init_dir()

        self.DATASETS = ['vost']
        self.TRAIN_TOTAL_STEPS = 20000
        self.DATA_SEQ_LEN = 25
        self.TRAIN_LONG_TERM_MEM_GAP = 4 if not (hasattr(self, "NO_MEMORY_GAP") and self.NO_MEMORY_GAP) else 1
        self.MODEL_LINEAR_Q = False
        self.MODEL_IGNORE_TOKEN = True

        self.TRAIN_AUTO_RESUME = False
        self.PRETRAIN_FULL = True  # if False, load encoder only
        if self.MODEL_VOS == 'aot':
            self.PRETRAIN_MODEL = os.path.join('pretrain_models', 'R50_AOTL_PRE_YTB_DAV.pth')
        elif self.MODEL_VOS == 'deaot':
            self.PRETRAIN_MODEL = os.path.join('pretrain_models', 'R50_DeAOTL_PRE_YTB_DAV.pth')
        else:
            raise NotImplementedError