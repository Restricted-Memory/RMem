import torch
import torch.nn as nn

from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
from utils.tensor import bchw_2_lbc
from timm.models.layers import trunc_normal_


class AOT(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(
            encoder,
            frozen_bn=cfg.MODEL_FREEZE_BN,
            freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT,
            use_mask=cfg.USE_MASK,
        )
        self.encoder_projector = nn.Conv2d(
            cfg.MODEL_ENCODER_DIM[-1],
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            kernel_size=1,
        )

        self.LSTT = LongShortTermTransformer(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            linear_q=cfg.MODEL_LINEAR_Q,
            norm_inp=cfg.MODEL_NORM_INP,
            time_encode=cfg.TIME_ENCODE,
            gru_memory=cfg.GRU_MEMORY,
        )

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * \
            (cfg.MODEL_LSTT_NUM + 1) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else \
            cfg.MODEL_ENCODER_EMBEDDING_DIM

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS,
        )

        id_dim = cfg.MODEL_MAX_OBJ_NUM + 1
        if cfg.MODEL_IGNORE_TOKEN:
            id_dim = cfg.MODEL_MAX_OBJ_NUM + 2
        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(
                id_dim,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=17,
                stride=16,
                padding=8,
            )
        else:
            self.patch_wise_id_bank = nn.Conv2d(
                id_dim,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=16,
                stride=16,
                padding=0,
            )

        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)

        self.pos_generator = PositionEmbeddingSine(
            cfg.MODEL_ENCODER_EMBEDDING_DIM // 2,
            normalize=True,
        )

        self._init_weight()

        self.__var_losses = []

        self.use_temporal_pe = cfg.USE_TEMPORAL_POSITIONAL_EMBEDDING
        if self.cfg.USE_TEMPORAL_POSITIONAL_EMBEDDING:
            self.cur_pos_emb = nn.Parameter(torch.randn(1, cfg.MODEL_ENCODER_EMBEDDING_DIM) * 0.05)
            if self.cfg.TEMPORAL_POSITIONAL_EMBEDDING_SLOT_4:
                self.mem_pos_emb = nn.Parameter(torch.randn(4, cfg.MODEL_ENCODER_EMBEDDING_DIM) * 0.05)
            else:
                self.mem_pos_emb = nn.Parameter(torch.randn(2, cfg.MODEL_ENCODER_EMBEDDING_DIM) * 0.05)
            trunc_normal_(self.cur_pos_emb, std=.05)
            trunc_normal_(self.mem_pos_emb, std=.05)
        else:
            self.temporal_encoding = None

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def encode_image(self, img, mask=None):
        if "topdown" in self.cfg.MODEL_ENCODER:
            if hasattr(self.cfg, "USE_MASK") and self.cfg.USE_MASK:
                mask_d = mask.detach()
                if (mask_d.shape[1] == 1) and not torch.is_floating_point(mask_d) and not torch.is_complex(mask_d):
                    mask_d = torch.where(mask_d == 255, 0, mask_d)
                    mask_d = (mask_d > 0).float()
                elif (mask_d.shape[1] > 1) and torch.is_floating_point(mask_d):
                    mask_d = 1-mask_d[:, 0:1, ...]
                else:
                    raise Exception("mask is not expected !")
                xs, var_loss = self.encoder(img, mask_d)
            else:
                xs, var_loss = self.encoder(img)
            self.__var_losses.append(var_loss)
        else:
            xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def LSTT_forward(
        self,
        curr_embs,
        curr_id_emb=None,
        pos_emb=None,
        size_2d=(30, 30),
        temporal_encoding=None,
        is_outer_memory=False,
        outer_long_memories=None,
        outer_short_memories=None,
        save_atten_weights=False,
    ):
        curr_emb = bchw_2_lbc(curr_embs[-1])
        lstt_embs = self.LSTT(
            curr_emb,
            curr_id_emb,
            pos_emb,
            size_2d=size_2d,
            temporal_encoding=temporal_encoding,
            is_outer_memory=is_outer_memory,
            outer_long_memories=outer_long_memories,
            outer_short_memories=outer_short_memories,
            save_atten_weights=save_atten_weights,
        )
        return lstt_embs

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(
                self.cfg.MODEL_ENCODER_EMBEDDING_DIM,
                -1,
            ).permute(0, 1),
            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)

    def get_var_loss(self):
        if len(self.__var_losses) == 0:
            return 0
        var_loss = torch.stack(self.__var_losses).mean()
        self.__var_losses = []
        return var_loss

    def init_LSTT_memory(self, size_2d=(30, 30)):
        self.LSTT.init_memory(size_2d)

    def clear_LSTT_memory(self):
        self.LSTT.clear_memory()

    def update_short_term_memory(
            self,
            curr_id_emb,
            short_term_mem_skip,
            size_2d,
            is_update_long_memory,
    ):
        self.LSTT.update_short_memories(
            curr_id_emb=curr_id_emb,
            short_term_mem_skip=short_term_mem_skip,
            size_2d=size_2d,
            is_update_long_memory=is_update_long_memory,
        )
