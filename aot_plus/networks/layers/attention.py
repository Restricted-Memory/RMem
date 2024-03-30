import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import DropOutLogit, DWConv2d


# Long-term attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head=8, dropout=0., use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.hidden_dim = d_model // num_head
        self.T = (d_model / num_head)**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V, is_return_attn_weight=False):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        if is_return_attn_weight:
            # Scale
            Q = Q / self.T

            # Multi-head
            Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 3, 0)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

            # Multiplication
            QK = Q @ K

            # Activation
            attn = torch.softmax(QK, dim=-1)

            # Dropouts
            attn = self.dropout(attn)

            # Weighted sum
            outputs = (attn @ V).permute(2, 0, 1, 3)
        else:
            dropout_p = self.drop_prob if self.training else 0.0
            Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            enable_mem_efficient = False if self.training else True
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=enable_mem_efficient):
                outputs = F.scaled_dot_product_attention(Q, K, V, None, dropout_p, is_causal=False)
            outputs = outputs.permute(2, 0, 1, 3)
            attn = None

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def silu(x):
    return x * torch.sigmoid(x)


class GatedPropagation(nn.Module):
    def __init__(self,
                 d_qk,
                 d_vu,
                 num_head=8,
                 dropout=0.,
                 use_linear=True,
                 d_att=None,
                 use_dis=False,
                 qk_chunks=1,
                 max_mem_len_ratio=-1,
                 top_k=-1,
                 expand_ratio=2.):
        super().__init__()
        expand_ratio = expand_ratio
        self.expand_d_vu = int(d_vu * expand_ratio)
        self.d_vu = d_vu
        self.d_qk = d_qk
        self.num_head = num_head
        self.use_dis = use_dis
        self.qk_chunks = qk_chunks
        self.max_mem_len_ratio = float(max_mem_len_ratio)
        self.top_k = top_k

        self.hidden_dim = self.expand_d_vu // num_head
        self.d_att = d_qk // num_head if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_linear = use_linear
        self.d_middle = self.d_att * self.num_head

        if use_linear:
            self.linear_QK = nn.Linear(d_qk, self.d_middle)
            half_d_vu = self.hidden_dim * num_head // 2
            self.linear_V1 = nn.Linear(d_vu // 2, half_d_vu)
            self.linear_V2 = nn.Linear(d_vu // 2, half_d_vu)
            self.linear_U1 = nn.Linear(d_vu // 2, half_d_vu)
            self.linear_U2 = nn.Linear(d_vu // 2, half_d_vu)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout

        self.dw_conv = DWConv2d(self.expand_d_vu)
        self.projection = nn.Linear(self.expand_d_vu, d_vu)

        self._init_weight()

    def forward(self, Q, K, V, U, size_2d, is_return_attn_weight=False):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        l, bs, _ = Q.size()

        # Linear projections
        if self.use_linear:
            Q = K = self.linear_QK(Q)

            def cat(X1, X2):
                if num_head > 1:
                    X1 = X1.view(-1, bs, num_head, hidden_dim // 2)
                    X2 = X2.view(-1, bs, num_head, hidden_dim // 2)
                    X = torch.cat([X1, X2],
                                  dim=-1).view(-1, bs, num_head * hidden_dim)
                else:
                    X = torch.cat([X1, X2], dim=-1)
                return X

            V1, V2 = torch.split(V, self.d_vu // 2, dim=-1)
            V1 = self.linear_V1(V1)
            V2 = self.linear_V2(V2)
            V = silu(cat(V1, V2))

            U1, U2 = torch.split(U, self.d_vu // 2, dim=-1)
            U1 = self.linear_U1(U1)
            U2 = self.linear_U2(U2)
            U = silu(cat(U1, U2))

        if is_return_attn_weight:
            # Scale
            Q = Q / self.T

            # Multi-head
            Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 3, 0)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

            # Multiplication
            QK = Q @ K

            # Activation
            attn = torch.softmax(QK, dim=-1)

            # Dropouts
            attn = self.dropout(attn)

            # Weighted sum
            outputs = (attn @ V).permute(2, 0, 1, 3)

        else:
            dropout_p = self.drop_prob if self.training else 0.0
            Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            enable_mem_efficient = False if self.training else True
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=enable_mem_efficient):
                outputs = F.scaled_dot_product_attention(Q, K, V, None, dropout_p, is_causal=False)
            outputs = outputs.permute(2, 0, 1, 3)
            attn = None
        # Restore shape
        outputs = outputs.reshape(l, bs, -1) * U

        outputs = self.dw_conv(outputs, size_2d)
        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class LocalGatedPropagation(nn.Module):
    def __init__(self,
                 d_qk,
                 d_vu,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 dilation=1,
                 use_linear=True,
                 enable_corr=True,
                 d_att=None,
                 use_dis=False,
                 expand_ratio=2.):
        super().__init__()
        expand_ratio = expand_ratio
        self.expand_d_vu = int(d_vu * expand_ratio)
        self.d_qk = d_qk
        self.d_vu = d_vu
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.hidden_dim = self.expand_d_vu // num_head
        self.d_att = d_qk // num_head if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_dis = use_dis

        self.d_middle = self.d_att * self.num_head
        self.use_linear = use_linear
        if use_linear:
            self.linear_QK = nn.Conv2d(d_qk, self.d_middle, kernel_size=1)
            self.linear_V = nn.Conv2d(d_vu,
                                      self.expand_d_vu,
                                      kernel_size=1,
                                      groups=2)
            self.linear_U = nn.Conv2d(d_vu,
                                      self.expand_d_vu,
                                      kernel_size=1,
                                      groups=2)

        self.relative_emb_k = nn.Conv2d(self.d_middle,
                                        num_head * self.window_size *
                                        self.window_size,
                                        kernel_size=1,
                                        groups=num_head)

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.dw_conv = DWConv2d(self.expand_d_vu)
        self.projection = nn.Linear(self.expand_d_vu, d_vu)

        self.dropout = nn.Dropout(dropout)

        self.drop_prob = dropout

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v, u, size_2d):
        n, c, h, w = v.size()
        hidden_dim = self.hidden_dim

        if self.use_linear:
            q = k = self.linear_QK(q)
            v = silu(self.linear_V(v))
            u = silu(self.linear_U(u))
            if self.num_head > 1:
                v = v.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(0, 2, 1, 3, 4).reshape(n, -1, h, w)
                u = u.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(4, 0, 2, 1, 3).reshape(h * w, n, -1)
            else:
                u = u.permute(2, 3, 0, 1).reshape(h * w, n, -1)

        if self.qk_mask is not None and (h, w) == self.last_size_2d:
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones((1, 1, h, w), device=v.device).float()
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w)
            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        # Scale
        q = q / self.T

        q = q.view(-1, self.d_att, h, w)
        k = k.view(-1, self.d_att, h, w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size, h * w)
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, self.d_att,
                self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(
                n, self.num_head, self.window_size * self.window_size, h * w)
        if self.use_dis:
            qk = 2 * qk - self.pad_and_unfold(
                k.pow(2).sum(dim=1, keepdim=True)).view(
                    n, self.num_head, self.window_size * self.window_size,
                    h * w)

        qk = qk + relative_emb

        qk -= qk_mask * 1e+8 if qk.dtype == torch.float32 else qk_mask * 1e+4

        local_attn = torch.softmax(qk, dim=2)

        local_attn = self.dropout(local_attn)

        global_attn = self.local2global(local_attn, h, w)

        agg_value = (global_attn @ v.transpose(-2, -1)).permute(
            2, 0, 1, 3).reshape(h * w, n, -1)

        output = agg_value * u

        output = self.dw_conv(output, size_2d)
        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None and (height,
                                            width) == self.last_size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=local_attn.device),
                torch.arange(0, pad_width, device=local_attn.device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=local_attn.device),
                torch.arange(0, width, device=local_attn.device)
            ])

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <=
                                                             self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height,
                                         pad_width)
            self.local_mask = local_mask

        global_attn = torch.zeros(
            (batch_size, self.num_head, height * width, pad_height, pad_width),
            device=local_attn.device)
        global_attn[local_mask.expand(batch_size, self.num_head,
                                      -1, -1, -1)] = local_attn.transpose(
                                          -1, -2).reshape(-1)
        global_attn = global_attn[:, :, :, self.max_dis:-self.max_dis,
                                  self.max_dis:-self.max_dis].reshape(
                                      batch_size, self.num_head,
                                      height * width, height * width)

        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                  mode='constant',
                  value=0)
        x = F.unfold(x,
                     kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1),
                     dilation=self.dilation)
        return x
