import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'pre_ytb'

        self.init_dir()

        pretrain_stage = 'PRE'
        pretrain_ckpt = 'save_step_100000.pth'
        self.DATA_SEQ_LEN = 10
        self.TRAIN_LONG_TERM_MEM_GAP = 4
        self.TRAIN_TOTAL_STEPS = 80000
        self.MODEL_LINEAR_Q = True
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                           self.EXP_NAME, pretrain_stage,
                                           'ema_ckpt', pretrain_ckpt)
