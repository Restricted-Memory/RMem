from typing import Iterable, List
import torch.nn.functional as F
from torch import nn
import torch

from networks.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d
from networks.layers.attention import MultiheadAttention, GatedPropagation, LocalGatedPropagation, silu
from utils.tensor import lbc_2_bchw, bchw_2_lbc
import numpy as np
from networks.debug import debug
# import random


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn' and groups != 1:
        return GroupNorm1D(indim, groups)
    elif type == 'gn' and groups == 1:
        return nn.GroupNorm(groups, indim)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim+hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def init_hidden(batch_size, hidden_dim, input_size, dtype):
        height, width = input_size
        return (torch.autograd.Variable(torch.zeros(batch_size, hidden_dim, height, width)).type(dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRUCellOutput(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, output_dim) -> None:
        super(ConvGRUCellOutput, self).__init__()
        self.conv_gru_cell = ConvGRUCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.output_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=1,
        )
    def forward(self, input_tensor, h_cur):
        h_next = self.conv_gru_cell(input_tensor, h_cur)
        return h_next, self.output_conv(h_next)


def mean_gru(a, b):
    out = (a + b) / 2
    return out, out


atten_condenser = MultiheadAttention(
    d_model=256,
    num_head=8,
    use_linear=False,
)
atten_condenser.projection = nn.Identity()

class LongShortTermTransformer(nn.Module):
    def __init__(
        self,
        num_layers=2,
        d_model=256,
        self_nhead=8,
        att_nhead=8,
        dim_feedforward=1024,
        emb_dropout=0.,
        droppath=0.1,
        lt_dropout=0.,
        st_dropout=0.,
        droppath_lst=False,
        droppath_scaling=False,
        activation="gelu",
        return_intermediate=False,
        intermediate_norm=True,
        final_norm=True,
        linear_q=False,
        norm_inp=False,
        time_encode=False,
        gru_memory=False,
    ):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        self.gru_memory = gru_memory

        layers: List[SimplifiedTransformerBlock] = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                SimplifiedTransformerBlock(
                    d_model, self_nhead, att_nhead,
                    dim_feedforward, droppath_rate,
                    activation,
                    linear_q=linear_q,
                    time_encode=time_encode,
                    gru_memory=gru_memory,
                ))
        self.layers: Iterable[SimplifiedTransformerBlock] = nn.ModuleList(
            layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

        self.clear_memory()

    def forward(
        self,
        tgt,
        curr_id_emb=None,
        self_pos=None,
        size_2d=None,
        temporal_encoding=None,
        is_outer_memory=False,
        outer_long_memories=None,
        outer_short_memories=None,
        save_atten_weights=False,
    ):

        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            if is_outer_memory:
                output, memories = layer(
                    output,
                    outer_long_memories[idx],
                    outer_short_memories[idx],
                    curr_id_emb=curr_id_emb,
                    self_pos=self_pos,
                    size_2d=size_2d,
                    temporal_encoding=temporal_encoding,
                    save_atten_weights=save_atten_weights,
                )
            else:
                output, memories = layer(
                    output,
                    self.long_term_memories[idx] if
                    self.long_term_memories is not None else None,
                    self.short_term_memories[idx] if
                    self.short_term_memories is not None else None,
                    curr_id_emb=curr_id_emb,
                    self_pos=self_pos,
                    size_2d=size_2d,
                    temporal_encoding=temporal_encoding,
                    save_atten_weights=save_atten_weights,
                )
            # memories : [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]]

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            if not is_outer_memory:
                self.lstt_curr_memories, self.lstt_long_memories, self.lstt_short_memories = zip(
                    *intermediate_memories)
            return intermediate

        return output

    def update_short_memories(
        self,
        curr_id_emb,
        short_term_mem_skip,
        size_2d,
        is_update_long_memory,
    ):
        lstt_curr_memories_2d = []
        for layer_idx in range(len(self.lstt_curr_memories)):
            curr_v = self.lstt_curr_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_V(
                curr_v + curr_id_emb)
            self.lstt_curr_memories[layer_idx][1] = curr_v

            curr_v = self.lstt_short_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_VMem(
                curr_v + curr_id_emb)
            self.lstt_short_memories[layer_idx][1] = curr_v

            lstt_curr_memories_2d.append([
                self.lstt_short_memories[layer_idx][0],
                self.lstt_short_memories[layer_idx][1],
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        for temp in self.short_term_memories_list[0]:
            for x in temp:
                x.cpu()
        self.short_term_memories_list = self.short_term_memories_list[
            -short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if is_update_long_memory:
            self.update_long_term_memory(
                self.lstt_curr_memories,
            )

    def update_long_term_memory(
        self,
        new_long_term_memories,
    ):
        updated_long_term_memories = []
        max_size = 48840
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(
                new_long_term_memory,
                last_long_term_memory,
            ):
                new_mem = torch.cat([last_e, new_e[None, ...]], dim=0)
                updated_e.append(new_mem)
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def restrict_long_memories(
        self,
        former_memory_len,
        latter_memory_len,
        use_atten_weight=False,
        long_memories_indexes=[],
        foreground_proba : torch.Tensor=None,
    ):
        to_drop_idx = former_memory_len
        if self.gru_memory:
            to_drop_idx += 1
        if use_atten_weight:
            # record_attn_weight [HW, T]
            # record_attn_weight.sum(dim=-1) = [1, 1, ... 1]
            attn_weight = torch.stack([
                self.layers[0].record_attn_weight,
                self.layers[0].record_attn_weight,
                # self.layers[1].record_attn_weight,
                # self.layers[2].record_attn_weight,
            ]).mean(dim=0)
            if foreground_proba is not None:
                foreground_proba = foreground_proba.squeeze().flatten().unsqueeze(-1)
                print(f"{foreground_proba.size() = }")
                attn_weight = attn_weight * foreground_proba
            attn_weight = attn_weight.sum(dim=0)
            attn_weight = attn_weight / attn_weight.sum()
            attn_weight = attn_weight.cpu()
            print(f"{attn_weight = }")

            # moving mean
            attn_weight_dict = {
                long_memories_indexes[i]: attn
                for i, attn in enumerate(attn_weight)
            }
            # print(f"{attn_weight_dict = }")
            last_attn_weight_dict = self.stored_attn_weight_dict
            moving_mean_factor = 0.8
            print(f"{moving_mean_factor = } x {last_attn_weight_dict = }")
            attn_weight_dict = {
                frame_idx:
                    (1-moving_mean_factor)*last_attn_weight_dict[frame_idx] + moving_mean_factor*attn
                    if frame_idx in last_attn_weight_dict else attn
                for frame_idx, attn in attn_weight_dict.items()
            }
            # print(f"{attn_weight_dict = }")
            self.stored_attn_weight_dict = attn_weight_dict
            for i, _ in enumerate(attn_weight):
                attn_weight[i] = attn_weight_dict[long_memories_indexes[i]]
            print(f"{attn_weight = }")

            # UCB
            frame_times = {
                mem_idx: 1
                for mem_idx in long_memories_indexes
            }
            # print(f"{self.stored_frame_times = }")
            frame_times = {
                mem_idx: (time + self.stored_frame_times[mem_idx]) if mem_idx in self.stored_frame_times else time
                for mem_idx, time in frame_times.items()
            }
            # print(f"{frame_times = }")
            self.stored_frame_times = frame_times
            frame_times_np = torch.Tensor([
                frame_times[mem_idx]
                for mem_idx in long_memories_indexes[:-1]
            ])
            print(f"{frame_times_np = }")
            frame_times_np[0] = len(frame_times_np)
            if (self.gru_memory) and len(frame_times_np) > 1:
                frame_times_np[1] = len(frame_times_np)
            add_item = 8
            mul_item = 1.5
            frame_times_param = mul_item * torch.sqrt(torch.log(frame_times_np.sum()) / (frame_times_np + add_item))
            print(f"{frame_times_param = }  +  {add_item = }  *  {mul_item = }")
            attn_weight = attn_weight + frame_times_param
            print(f"{attn_weight = }")

            # print(f"{attn_weight = }")
            ignore_former_size = 1
            if self.gru_memory:
                ignore_former_size += 1
            attn_weight_remove_0 = attn_weight[ignore_former_size:]
            if attn_weight_remove_0.size(0) > 0:
                to_drop_idx = torch.argmin(attn_weight_remove_0).item()
                to_drop_idx += ignore_former_size
        print(f"{to_drop_idx = }")
        is_drop = False
        for layer_idx in range(len(self.layers)):
            memory_k_v = self.long_term_memories[layer_idx]
            for i in range(len(memory_k_v)):
                mem = memory_k_v[i]
                if mem.size(0) > (former_memory_len + latter_memory_len):
                    is_drop = True
                    if self.gru_memory:
                        gru = self.layers[layer_idx].memory_grus[i]
                        # gru = mean_gru
                        hidden_state = self.long_term_memory_hidden_states[layer_idx][i]
                        size_2d = self.long_term_memory_hidden_states[0][0].size()[2:]
                        gru_input = lbc_2_bchw(mem[to_drop_idx, ...], size_2d)
                        hidden_state, gru_output = gru(gru_input, hidden_state)
                        gru_output = bchw_2_lbc(gru_output)
                        new_mem = torch.cat(
                            [mem[0:1, ...], gru_output[None, ...], mem[2:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                        self.long_term_memory_hidden_states[layer_idx][i] = hidden_state
                    else:
                        new_mem = torch.cat(
                            [mem[0:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                    self.long_term_memories[layer_idx][i] = new_mem
        if is_drop:
            long_memories_indexes.remove(long_memories_indexes[to_drop_idx])

    def init_memory(self, size_2d=(30, 30)):
        self.long_term_memories = self.lstt_long_memories
        self.short_term_memories_list = [self.lstt_short_memories]
        self.short_term_memories = self.lstt_short_memories
        self.stored_attn_weight_dict = {}
        self.stored_frame_times = {}
        if self.gru_memory:
            l, b, c = self.short_term_memories[0][0].size()
            dtype = self.short_term_memories[0][0].dtype
            device = self.short_term_memories[0][0].device
            self.long_term_memory_hidden_states = [
                [
                    ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device),
                    ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device),
                ] for _ in range(len(self.layers))
            ]

    def clear_memory(self):
        self.lstt_curr_memories = None
        self.lstt_long_memories = None
        self.lstt_short_memories = None

        self.short_term_memories_list = []
        self.short_term_memories = None
        self.long_term_memories = None
        self.long_term_memory_hidden_states = None


class SimplifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        self_nhead,
        att_nhead,
        dim_feedforward=1024,
        droppath=0.1,
        activation="gelu",
        linear_q=False,
        time_encode=False,
        gru_memory=False,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_QMem = nn.Linear(d_model, d_model)
        self.linear_VMem = nn.Linear(d_model, d_model)
        if not linear_q:
            self.norm4 = _get_norm(d_model)

        self.linear_KMem = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(
            d_model,
            att_nhead,
            use_linear=False,
        )

        self.short_term_attn = MultiheadAttention(
            d_model,
            att_nhead,
            use_linear=False,
        )

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self.linear_q = linear_q
        self._init_weight()

        if time_encode:
            self.Q_time_encode = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.ReLU(),
                nn.Linear(in_features=d_model, out_features=d_model),
            )
            self.K_time_encode = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.ReLU(),
                nn.Linear(in_features=d_model, out_features=d_model),
            )
        if gru_memory:
            self.memory_grus = nn.ModuleList([
                ConvGRUCellOutput(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    kernel_size=(2, 2),
                    bias=True,
                    output_dim=d_model,
                ),
                ConvGRUCellOutput(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    kernel_size=(1, 1),
                    bias=True,
                    output_dim=d_model,
                ),
            ])
    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        long_term_memory=None,
        short_term_memory=None,
        curr_id_emb=None,
        self_pos=None,
        size_2d=(30, 30),
        temporal_encoding=None,
        save_atten_weights=False,
    ):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = curr_Q

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = self.linear_V(curr_V + curr_id_emb)
            local_K = global_K
            local_V = global_V
            global_K = global_K[None, ...]
            global_V = global_V[None, ...]
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory


        if temporal_encoding is not None:
            T, token_num, bs, embed_dim = global_K.shape
            cur_pos_emb, mem_pos_emb = temporal_encoding[0:1], temporal_encoding[1:]

            max_T = 4

            if T <= mem_pos_emb.size(0):
                mem_pos_emb = mem_pos_emb[:T]

            if T == 1:
                flatten_global_K = global_K + mem_pos_emb[0].view(1 ,1, 1, embed_dim)
            else:
                interpolated_mem_pe = mem_pos_emb.clone().permute(1, 0).view(1, embed_dim, -1)
                if T <= max_T:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='linear', align_corners=True)
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")
                else:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=max_T, mode='linear', align_corners=True) # 1 * embed_dim * max_T
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='nearest') # 1 * embed_dim * T
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                    # interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='nearest-exact') # 1 * embed_dim * T
                    # debug(f"nearest:  {interpolated_mem_pe[0, 0, :] = }")
                    # interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='linear', align_corners=True) # 1 * embed_dim * T
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")

                interpolated_mem_pe = interpolated_mem_pe.view(embed_dim, T).permute(1, 0).contiguous()

                flatten_global_K = global_K.view(T, token_num, bs, embed_dim) + interpolated_mem_pe.view(T, 1, 1, embed_dim)

            flatten_global_K = flatten_global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q + cur_pos_emb.view(1, 1, embed_dim)
        else:    
            flatten_global_K = global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q
        flatten_global_V = global_V.flatten(0, 1)

        tgt2, attn = self.long_term_attn(
            curr_Q_add_time, flatten_global_K, flatten_global_V,
            is_return_attn_weight=save_atten_weights,
        )
        if save_atten_weights:
            bs, head, hw, thw = attn.size()
            self.record_T = thw // hw
            attn = attn.view((bs, head, hw, self.record_T, hw))
            attn = attn.mean(dim=1) # bs, hw, T, hw
            assert attn.size(0) == 1 # only for evaluation
            attn = attn.squeeze(0) # hw, T, hw
            self.record_attn_weight = attn.sum(dim=2)
            self.attn_values, self.attn_indices = \
                attn.detach().view((hw, thw)).topk(32, dim=-1)
            self.attn_values, self.attn_indices = \
                self.attn_values.cpu().numpy(), self.attn_indices.cpu().numpy()
            self.attn_indices = np.unravel_index(self.attn_indices, (self.record_T, hw))

        if self.linear_q:
            tgt3 = self.short_term_attn(
                local_Q,
                torch.cat((local_K, curr_K), 0),
                torch.cat((local_V, curr_V), 0),
            )[0]
        else:
            tgt3, short_attn = self.short_term_attn(
                local_Q,
                self.norm4(local_K + curr_K),
                self.norm4(local_V + curr_V),
                is_return_attn_weight=save_atten_weights,
            )
        if save_atten_weights:
            # bs, head, hw, hw
            short_attn = short_attn.mean(dim=1) # bs, hw, hw
            assert short_attn.size(0) == 1 # only for evaluation
            short_attn = short_attn.squeeze(0) # hw, hw
            self.short_attn_values, self.short_attn_indices = \
                short_attn.detach().topk(32, dim=-1)
            self.short_attn_values, self.short_attn_indices = \
                self.short_attn_values.cpu().numpy(), self.short_attn_indices.cpu().numpy()

        _tgt3 = tgt3

        local_K = self.linear_QMem(_tgt3)
        local_V = _tgt3
        if curr_id_emb is not None:
            local_V = self.linear_VMem(local_V + curr_id_emb)

        tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [
            [curr_K, curr_V], [global_K, global_V],
            [local_K, local_V],
        ]

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class DualBranchGPM(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        # self.mask_token = nn.Parameter(torch.randn([1, 1, d_model]))

        block = GatedPropagationModule
        self.gru_memory = False

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                block(d_model,
                      self_nhead,
                      att_nhead,
                      dim_feedforward,
                      droppath_rate,
                      lt_dropout,
                      st_dropout,
                      droppath_lst,
                      activation,
                      layer_idx=idx))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model * 2, type='gn', groups=2)
            for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

        self.clear_memory()

    def forward(self,
                tgt,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None,
                temporal_encoding=None,
                is_outer_memory=False,
                outer_long_memories=None,
                outer_short_memories=None,
                save_atten_weights=False,
            ):

        output = self.emb_dropout(tgt)

        # output = mask_out(output, self.mask_token, 0.15, self.training)

        intermediate = []
        intermediate_memories = []
        output_id = None

        for idx, layer in enumerate(self.layers):
            output, output_id, memories = layer(
                output,
                output_id,
                self.long_term_memories[idx]
                if self.long_term_memories is not None else None,
                self.short_term_memories[idx]
                if self.short_term_memories is not None else None,
                curr_id_emb=curr_id_emb,
                self_pos=self_pos,
                size_2d=size_2d,
                temporal_encoding=temporal_encoding,
                save_atten_weights=save_atten_weights,
            )

            cat_output = torch.cat([output, output_id], dim=2)

            if self.return_intermediate:
                intermediate.append(cat_output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                cat_output = self.decoder_norms[-1](cat_output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(cat_output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            self.lstt_curr_memories, self.lstt_long_memories, self.lstt_short_memories = zip(
                *intermediate_memories)
            return intermediate

        return cat_output

    def update_short_memories(
        self,
        curr_id_emb,
        short_term_mem_skip,
        size_2d,
        is_update_long_memory,
    ):
        lstt_curr_memories_2d = []
        for layer_idx in range(len(self.lstt_curr_memories)):
            curr_k, curr_v, curr_id_k, curr_id_v = self.lstt_curr_memories[
                layer_idx]
            curr_id_k, curr_id_v = self.layers[
                layer_idx].fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb)
            self.lstt_curr_memories[layer_idx][2], self.lstt_curr_memories[layer_idx][
                3] = curr_id_k, curr_id_v
            local_curr_id_k = seq_to_2d(
                curr_id_k, size_2d) if curr_id_k is not None else None
            local_curr_id_v = seq_to_2d(curr_id_v, size_2d)
            lstt_curr_memories_2d.append([
                seq_to_2d(curr_k, size_2d),
                seq_to_2d(curr_v, size_2d), local_curr_id_k,
                local_curr_id_v
            ])
        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if is_update_long_memory:
            self.update_long_term_memory(
                self.lstt_curr_memories,
            )

    def update_long_term_memory(
        self,
        new_long_term_memories,
    ):
        updated_long_term_memories = []
        max_size = 48840
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(
                new_long_term_memory,
                last_long_term_memory,
            ):
                if new_e is None or last_e is None:
                    new_mem = None
                else:
                    new_mem = torch.cat([last_e, new_e[None, ...]], dim=0)
                updated_e.append(new_mem)
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def restrict_long_memories(
        self,
        former_memory_len,
        latter_memory_len,
        use_atten_weight=False,
        long_memories_indexes=[],
        foreground_proba : torch.Tensor=None,
    ):
        to_drop_idx = former_memory_len
        if self.gru_memory:
            to_drop_idx += 1
        if use_atten_weight:
            # record_attn_weight [HW, T]
            # record_attn_weight.sum(dim=-1) = [1, 1, ... 1]
            attn_weight = torch.stack([
                self.layers[0].record_attn_weight,
                self.layers[0].record_attn_weight,
                # self.layers[1].record_attn_weight,
                # self.layers[2].record_attn_weight,
            ]).mean(dim=0)
            if foreground_proba is not None:
                foreground_proba = foreground_proba.squeeze().flatten().unsqueeze(-1)
                print(f"{foreground_proba.size() = }")
                attn_weight = attn_weight * foreground_proba
            attn_weight = attn_weight.sum(dim=0)
            attn_weight = attn_weight / attn_weight.sum()
            attn_weight = attn_weight.cpu()
            print(f"{attn_weight = }")

            # moving mean
            attn_weight_dict = {
                long_memories_indexes[i]: attn
                for i, attn in enumerate(attn_weight)
            }
            # print(f"{attn_weight_dict = }")
            last_attn_weight_dict = self.stored_attn_weight_dict
            moving_mean_factor = 0.8
            print(f"{moving_mean_factor = } x {last_attn_weight_dict = }")
            attn_weight_dict = {
                frame_idx:
                    (1-moving_mean_factor)*last_attn_weight_dict[frame_idx] + moving_mean_factor*attn
                    if frame_idx in last_attn_weight_dict else attn
                for frame_idx, attn in attn_weight_dict.items()
            }
            # print(f"{attn_weight_dict = }")
            self.stored_attn_weight_dict = attn_weight_dict
            for i, _ in enumerate(attn_weight):
                attn_weight[i] = attn_weight_dict[long_memories_indexes[i]]
            print(f"{attn_weight = }")

            # UCB
            frame_times = {
                mem_idx: 1
                for mem_idx in long_memories_indexes
            }
            # print(f"{self.stored_frame_times = }")
            frame_times = {
                mem_idx: (time + self.stored_frame_times[mem_idx]) if mem_idx in self.stored_frame_times else time
                for mem_idx, time in frame_times.items()
            }
            # print(f"{frame_times = }")
            self.stored_frame_times = frame_times
            frame_times_np = torch.Tensor([
                frame_times[mem_idx]
                for mem_idx in long_memories_indexes[:-1]
            ])
            print(f"{frame_times_np = }")
            frame_times_np[0] = len(frame_times_np)
            if (self.gru_memory) and len(frame_times_np) > 1:
                frame_times_np[1] = len(frame_times_np)
            add_item = 8
            mul_item = 1.5
            frame_times_param = mul_item * torch.sqrt(torch.log(frame_times_np.sum()) / (frame_times_np + add_item))
            print(f"{frame_times_param = }  +  {add_item = }  *  {mul_item = }")
            attn_weight = attn_weight + frame_times_param
            print(f"{attn_weight = }")

            # print(f"{attn_weight = }")
            ignore_former_size = 1
            if self.gru_memory:
                ignore_former_size += 1
            attn_weight_remove_0 = attn_weight[ignore_former_size:]
            if attn_weight_remove_0.size(0) > 0:
                to_drop_idx = torch.argmin(attn_weight_remove_0).item()
                to_drop_idx += ignore_former_size
        print(f"{to_drop_idx = }")
        is_drop = False
        for layer_idx in range(len(self.layers)):
            memory_k_v = self.long_term_memories[layer_idx]
            for i in range(len(memory_k_v)):
                mem = memory_k_v[i]
                if mem is None:
                    continue
                if mem.size(0) > (former_memory_len + latter_memory_len):
                    is_drop = True
                    if self.gru_memory:
                        gru = self.layers[layer_idx].memory_grus[i]
                        # gru = mean_gru
                        hidden_state = self.long_term_memory_hidden_states[layer_idx][i]
                        size_2d = self.long_term_memory_hidden_states[0][0].size()[2:]
                        gru_input = lbc_2_bchw(mem[to_drop_idx, ...], size_2d)
                        hidden_state, gru_output = gru(gru_input, hidden_state)
                        gru_output = bchw_2_lbc(gru_output)
                        new_mem = torch.cat(
                            [mem[0:1, ...], gru_output[None, ...], mem[2:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                        self.long_term_memory_hidden_states[layer_idx][i] = hidden_state
                    else:
                        new_mem = torch.cat(
                            [mem[0:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                    self.long_term_memories[layer_idx][i] = new_mem
        if is_drop:
            long_memories_indexes.remove(long_memories_indexes[to_drop_idx])

    def init_memory(self, size_2d=(30, 30)):
        self.long_term_memories = self.lstt_long_memories
        self.short_term_memories_list = [self.lstt_short_memories]
        self.short_term_memories = self.lstt_short_memories
        self.stored_attn_weight_dict = {}
        self.stored_frame_times = {}

    def clear_memory(self):
        self.lstt_curr_memories = None
        self.lstt_long_memories = None
        self.lstt_short_memories = None

        self.short_term_memories_list = []
        self.short_term_memories = None
        self.long_term_memories = None


class GatedPropagationModule(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True,
                 max_local_dis=7,
                 layer_idx=0,
                 expand_ratio=2.):
        super().__init__()
        expand_ratio = expand_ratio
        expand_d_model = int(d_model * expand_ratio)
        self.expand_d_model = expand_d_model
        self.d_model = d_model
        self.att_nhead = att_nhead

        d_att = d_model // 2 if att_nhead == 1 else d_model // att_nhead
        self.d_att = d_att
        self.layer_idx = layer_idx

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, d_att * att_nhead + expand_d_model)
        self.linear_U = nn.Linear(d_model, expand_d_model)

        if layer_idx == 0:
            self.linear_ID_V = nn.Linear(d_model, expand_d_model)
        else:
            self.id_norm1 = _get_norm(d_model)
            self.linear_ID_V = nn.Linear(d_model * 2, expand_d_model)
            self.linear_ID_U = nn.Linear(d_model, expand_d_model)

        self.long_term_attn = GatedPropagation(d_qk=self.d_model,
                                    d_vu=self.d_model * 2,
                                    num_head=att_nhead,
                                    use_linear=False,
                                    dropout=lt_dropout,
                                    d_att=d_att,
                                    top_k=-1,
                                    expand_ratio=expand_ratio)

        enable_corr = False
        self.short_term_attn = LocalGatedPropagation(d_qk=self.d_model,
                                          d_vu=self.d_model * 2,
                                          num_head=att_nhead,
                                          dilation=local_dilation,
                                          use_linear=False,
                                          enable_corr=enable_corr,
                                          dropout=st_dropout,
                                          d_att=d_att,
                                          max_dis=max_local_dis,
                                          expand_ratio=expand_ratio)

        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.id_norm2 = _get_norm(d_model)
        self.self_attn = GatedPropagation(d_model * 2,
                               d_model * 2,
                               self_nhead,
                               d_att=d_att)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                tgt_id=None,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30),
                temporal_encoding=None,
                save_atten_weights=False,
            ):

        # Long Short-Term Attention
        _tgt = self.norm1(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(
            curr_QV, [self.d_att * self.att_nhead, self.expand_d_model], dim=2)
        curr_Q = curr_K = curr_QV[0]
        local_Q = seq_to_2d(curr_Q, size_2d)
        curr_V = silu(curr_QV[1])
        curr_U = self.linear_U(_tgt)

        if tgt_id is None:
            tgt_id = 0
            cat_curr_U = torch.cat(
                [silu(curr_U), torch.ones_like(curr_U)], dim=-1)
            curr_ID_V = None
        else:
            _tgt_id = self.id_norm1(tgt_id)
            curr_ID_V = _tgt_id
            curr_ID_U = self.linear_ID_U(_tgt_id)
            cat_curr_U = silu(torch.cat([curr_U, curr_ID_U], dim=-1))

        if curr_id_emb is not None:
            global_K, global_V = curr_K, curr_V
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)

            _, global_ID_V = self.fuse_key_value_id(None, curr_ID_V,
                                                    curr_id_emb)
            local_ID_V = seq_to_2d(global_ID_V, size_2d)
            global_K = global_K[None, ...]
            global_V = global_V[None, ...]
            global_ID_V = global_ID_V[None, ...]
        else:
            global_K, global_V, _, global_ID_V = long_term_memory
            local_K, local_V, _, local_ID_V = short_term_memory

        if temporal_encoding is not None:
            T, token_num, bs, embed_dim = global_K.shape
            cur_pos_emb, mem_pos_emb = temporal_encoding[0:1], temporal_encoding[1:]

            max_T = 4

            if T <= mem_pos_emb.size(0):
                mem_pos_emb = mem_pos_emb[:T]

            if T == 1:
                flatten_global_K = global_K + mem_pos_emb[0].view(1 ,1, 1, embed_dim)
            else:
                interpolated_mem_pe = mem_pos_emb.clone().permute(1, 0).view(1, embed_dim, -1)
                if T <= max_T:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='linear', align_corners=True)
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")
                else:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=max_T, mode='linear', align_corners=True) # 1 * embed_dim * max_T
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='nearest') # 1 * embed_dim * T
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                    # interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='nearest-exact') # 1 * embed_dim * T
                    # debug(f"nearest:  {interpolated_mem_pe[0, 0, :] = }")
                    # interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='linear', align_corners=True) # 1 * embed_dim * T
                    # debug(f"linear:  {interpolated_mem_pe[0, 0, :] = }")

                interpolated_mem_pe = interpolated_mem_pe.view(embed_dim, T).permute(1, 0).contiguous()

                flatten_global_K = global_K.view(T, token_num, bs, embed_dim) + interpolated_mem_pe.view(T, 1, 1, embed_dim)

            flatten_global_K = flatten_global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q + cur_pos_emb.view(1, 1, embed_dim)
        else:    
            flatten_global_K = global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q

        flatten_global_V = global_V.flatten(0, 1)
        flatten_global_ID_V = global_ID_V.flatten(0, 1)

        cat_global_V = torch.cat([flatten_global_V, flatten_global_ID_V], dim=-1)
        cat_local_V = torch.cat([local_V, local_ID_V], dim=1)

        cat_tgt2, attn = self.long_term_attn(curr_Q_add_time, flatten_global_K, cat_global_V,
                                          cat_curr_U, size_2d, is_return_attn_weight=save_atten_weights)
        if save_atten_weights:
            bs, head, hw, thw = attn.size()
            self.record_T = thw // hw
            attn = attn.view((bs, head, hw, self.record_T, hw))
            attn = attn.mean(dim=1) # bs, hw, T, hw
            assert attn.size(0) == 1 # only for evaluation
            attn = attn.squeeze(0) # hw, T, hw
            self.record_attn_weight = attn.sum(dim=2)
            self.attn_values, self.attn_indices = \
                attn.detach().view((hw, thw)).topk(32, dim=-1)
            self.attn_values, self.attn_indices = \
                self.attn_values.cpu().numpy(), self.attn_indices.cpu().numpy()
            self.attn_indices = np.unravel_index(self.attn_indices, (self.record_T, hw))

        cat_tgt3, short_attn = self.short_term_attn(local_Q, local_K, cat_local_V,
                                           cat_curr_U, size_2d)

        if save_atten_weights:
            # bs, head, hw, hw
            short_attn = short_attn.mean(dim=1) # bs, hw, hw
            assert short_attn.size(0) == 1 # only for evaluation
            short_attn = short_attn.squeeze(0) # hw, hw
            self.short_attn_values, self.short_attn_indices = \
                short_attn.detach().topk(32, dim=-1)
            self.short_attn_values, self.short_attn_indices = \
                self.short_attn_values.cpu().numpy(), self.short_attn_indices.cpu().numpy()

        tgt2, tgt_id2 = torch.split(cat_tgt2, self.d_model, dim=-1)
        tgt3, tgt_id3 = torch.split(cat_tgt3, self.d_model, dim=-1)

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
            tgt_id = tgt_id + self.droppath(tgt_id2 + tgt_id3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)
            tgt_id = tgt_id + self.lst_dropout(tgt_id2 + tgt_id3)

        # Self-attention
        _tgt = self.norm2(tgt)
        _tgt_id = self.id_norm2(tgt_id)
        q = k = v = u = torch.cat([_tgt, _tgt_id], dim=-1)

        cat_tgt2, _ = self.self_attn(q, k, v, u, size_2d)

        tgt2, tgt_id2 = torch.split(cat_tgt2, self.d_model, dim=-1)

        tgt = tgt + self.droppath(tgt2)
        tgt_id = tgt_id + self.droppath(tgt_id2)

        return tgt, tgt_id, [[curr_K, curr_V, None, curr_ID_V],
                             [global_K, global_V, None, global_ID_V],
                             [local_K, local_V, None, local_ID_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_K = None
        if value is not None:
            ID_V = silu(self.linear_ID_V(torch.cat([value, id_emb], dim=2)))
        else:
            ID_V = silu(self.linear_ID_V(id_emb))
        return ID_K, ID_V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
