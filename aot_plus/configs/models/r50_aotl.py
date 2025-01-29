from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.FORMER_MEM_LEN = 1
        self.LATTER_MEM_LEN = 8
        self.GRU_MEMORY = False
        self.FREEZE_AOT_EXCEPT_GRU = self.GRU_MEMORY and True
        gru_memory_text = f"_Gru_mem_{self.FORMER_MEM_LEN}_{self.LATTER_MEM_LEN}{'_Freeze' if self.FREEZE_AOT_EXCEPT_GRU else ''}" if self.GRU_MEMORY else ""
        self.TIME_ENCODE = False
        self.TIME_ENCODE_NORM = self.TIME_ENCODE and True
        time_encode_text = f"_Time_encode{'_norm' if self.TIME_ENCODE_NORM else ''}" if self.TIME_ENCODE else ""
        self.USE_TEMPORAL_POSITIONAL_EMBEDDING = True
        self.FREEZE_AOT_EXCEPT_TEMPORAL_EMB = self.USE_TEMPORAL_POSITIONAL_EMBEDDING and False
        self.TEMPORAL_POSITIONAL_EMBEDDING_SLOT_4 = self.USE_TEMPORAL_POSITIONAL_EMBEDDING and True
        temporal_pe_text = f"_Temp_pe{'_Freeze' if self.FREEZE_AOT_EXCEPT_TEMPORAL_EMB else ''}{'_Slot_4' if self.TEMPORAL_POSITIONAL_EMBEDDING_SLOT_4 else ''}" if self.USE_TEMPORAL_POSITIONAL_EMBEDDING else ""
        self.USE_MASK = False
        self.NO_LONG_MEMORY = False
        long_mem_text = "_No_long_mem" if self.NO_LONG_MEMORY else ""
        self.NO_MEMORY_GAP = False
        self.MODEL_ATT_HEADS = 2 if self.NO_MEMORY_GAP else self.MODEL_ATT_HEADS
        mem_gap_text = "_No_mem_gap" if self.NO_MEMORY_GAP else ""
        self.REVERSE_INFER = False
        self.REVERSE_LOSS = 0.4
        self.REVERSE_LOSS = self.REVERSE_LOSS / 4 if self.NO_MEMORY_GAP else self.REVERSE_LOSS
        reverse_infer_text = f"_Reverse_infer_detach_short_loss_{self.REVERSE_LOSS}" if self.REVERSE_INFER else ""
        self.MODEL_NAME = f'R50_AOTL{time_encode_text}{long_mem_text}{mem_gap_text}{reverse_infer_text}{gru_memory_text}'
        self.MODEL_NAME += temporal_pe_text

        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5