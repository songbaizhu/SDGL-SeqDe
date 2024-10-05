import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from graph_distribution import graph_constructor
from layer_distribution_mtgnn import *
import numpy as np
from STID_arch import STID
from layer_distribution_module import RevIN
from mlp import MultiLayerPerceptron


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            x = torch.einsum('vw, ncwl->ncvl', [A, x])
        elif len(A.size()) == 3:
            x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        else:
            raise ValueError("A size error")
        return x.contiguous()


class patch_nconv(torch.nn.Module):
    def __init__(self):
        super(patch_nconv, self).__init__()

    def forward(self, x, A):
        # x:B×C×N×P
        # A:B×P×N×N
        # return B×C×N×P
        x = torch.einsum("bcwp,bpvw->bcvp", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, seq_length=None, patch_len=None):
        super(gcn_module, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

        # 动态图
        self.patch_nconv = patch_nconv()
        self.seq_length = seq_length
        self.patch_len = patch_len

    def static(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

    def dynamic(self, x, A):
        # x:B×C×N×P
        # A:B×P×N×N
        B, C, N, P = x.shape

        rest_fea = None
        h = None
        out = []
        if P > A.shape[1]:
            rest_fea = x[..., :P - A.shape[1]]
            out.append(x[..., -A.shape[1]:])
            h = x[..., -A.shape[1]:]
        else:

            A = A[:, -P:, :, :]
            out.append(x)
            h = x

        for i in range(self.order):
            h = self.patch_nconv(h, A)
            out.append(h)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        if P > A.shape[1]:
            ho = torch.cat([rest_fea, ho], dim=-1)
        return F.dropout(ho, p=self.dropout, training=self.training)

    def forward(self, x, AL, AS, short_bool=True, smooth_depth=1):
        if short_bool == False:
            return self.static(x, AL)

        B, C, N, P = x.shape

        rest_fea = None
        h = None
        out = []
        if P > AS.shape[1]:
            rest_fea = x[..., :P - AS.shape[1]]
            out.append(x[..., -AS.shape[1]:])
            h = x[..., -AS.shape[1]:]
        else:
            AS = AS[:, -P:, :, :]
            out.append(x)
            h = x

        h_long = torch.einsum("bckj,ik->bcij", h, torch.matrix_power(AL, smooth_depth))
        h_short = h - h_long
        for i in range(self.order):
            h_long = self.nconv(h_long, AL)
            h_short = self.patch_nconv(h_short, AS)
            out.append(h_long + h_short)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        if rest_fea is not None:
            ho = torch.cat([rest_fea, ho], dim=-1)
        return F.dropout(ho, p=self.dropout, training=self.training)


class gcn_module_diff(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, seq_length=None, patch_len=None):
        super(gcn_module_diff, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

        # 动态图
        self.patch_nconv = patch_nconv()
        self.seq_length = seq_length
        self.patch_len = patch_len

    def static(self, x, support):
        out = [x]
        for k in range(1, self.order + 1):
            x = x - self.nconv(x, support)
            out.append(x)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Fusioncell(nn.Module):
    def __init__(self, channel):
        super(Fusioncell, self).__init__()
        self.conv = nn.Conv2d(2 * channel, channel, (1, 3), padding='same')

    def forward(self, x, mlp_part):
        # x B×C×N×T
        # mlp_part B×C×N×1

        T = x.shape[-1]
        x_rest = None
        if T > mlp_part.shape[-1]:
            x_rest = x[..., :T - mlp_part.shape[-1]]
            x = x[..., -mlp_part.shape[-1]:]
        else:
            mlp_part = mlp_part[..., -T:]
        rate = torch.cat([x, mlp_part], dim=1)
        rate = torch.sigmoid(self.conv(rate))
        out = rate * x + (1 - rate) * mlp_part
        if x_rest is not None:
            out = torch.cat([x_rest, out], dim=-1)
        return out


class Fusionshort(nn.Module):
    def __init__(self, channel):
        super(Fusionshort, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, (1, 1), padding='same')
        self.conv2 = nn.Conv2d(channel, channel, (1, 1), padding='same')

    def forward(self, x, x_diff, x_short):
        # x B×C×N×T
        x_diff = F.relu(self.conv1(x_diff))
        rate = torch.sigmoid(self.conv2(x_diff))
        out = rate * x + (1 - rate) * x_short
        return out


class Tiem_conv(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_factor=1):
        super(Tiem_conv, self).__init__()
        self.filter = dilated_inception(in_channel, out_channel, dilation_factor=dilation_factor)
        self.gate = dilated_inception(in_channel, out_channel, dilation_factor=dilation_factor)

    def forward(self, x):
        f = self.filter(x)
        f = torch.tanh(f)
        g = self.gate(x)
        g = torch.sigmoid(g)
        return f * g


class gwnet_part1(nn.Module):

    def __init__(self, device, short_bool, smooth_depth, args, distribute_data=None):
        super(gwnet_part1, self).__init__()
        self.device = device
        self.short_bool = short_bool
        self.smooth_depth = smooth_depth
        num_nodes = args.num_nodes
        dropout = args.dropout
        self.checkpoint = args.checkpoint

        in_dim = args.in_dim
        residual_channels = args.residual_channels
        # dilation_channels = args.nhid
        skip_channels = args.skip_channels
        end_channels = args.end_channels

        layers = args.layers

        self.dropout = dropout

        self.layers = args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj

        self.time_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.fusion_t_mlp = nn.ModuleList()
        self.fusion_g_mlp = nn.ModuleList()
        self.fusion_start_mlp = Fusioncell(residual_channels)

        self.nodes = num_nodes
        self.seq_length = args.lag
        self.patch_len = args.patch_len
        self.patch_num = self.seq_length // self.patch_len

        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                     out_channels=residual_channels,
                                     kernel_size=(1, self.patch_len), stride=(1, self.patch_len))

        kernel_size = 7  # 12 #

        dilation_exponential = args.dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        rf_size_i = 1
        new_dilation = 1
        num = 1

        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.time_convs.append(Tiem_conv(num * residual_channels, residual_channels, dilation_factor=new_dilation))
            self.fusion_t_mlp.append(Fusioncell(residual_channels))
            self.fusion_g_mlp.append(Fusioncell(residual_channels))
            self.residual_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(
                                                 1, max(self.patch_num, self.receptive_field) - rf_size_j + 1)))

            if self.gcn_bool:
                self.gconv.append(gcn_module(residual_channels, residual_channels, dropout, support_len=1, order=2,
                                             seq_length=self.patch_num, patch_len=self.patch_len))

            self.norm.append(LayerNorm(
                (skip_channels, num_nodes, 1),
                elementwise_affine=True))

            new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=args.output_len,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                               kernel_size=(1, self.seq_length),
                               bias=True)
        self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                               kernel_size=(1, max(self.patch_num, self.receptive_field) - self.receptive_field + 1),
                               bias=True)

        self.idx = torch.arange(self.nodes).to(device)

        self.graph_construct = graph_constructor(args.in_dim, self.nodes, args.embed_dim, device,
                                                 patch_len=self.patch_len)

        self.use_RevIN = args.use_RevIN
        if args.use_RevIN:
            self.revin = RevIN(args.num_nodes)

    def fun1(self, i, x, x_short, x_diff, skip, skip_short, skip_diff, new_supports, dynamic_adj_pos, dynamic_adj_neg,
             mlp_component):
        residual = x

        x = self.time_convs[i](x)

        x = self.fusion_t_mlp[i](x, mlp_component)
        x = F.dropout(x, self.dropout, training=self.training)

        skip = self.norm[i](self.skip_convs[i](x) + skip, self.idx)

        if self.gcn_bool:
            x = self.gconv[i](x, new_supports, dynamic_adj_pos, self.short_bool, 1)
        else:
            x = self.residual_convs[i](x)

        x = self.fusion_g_mlp[i](x, mlp_component)

        x = x + residual[:, :, :, -x.size(3):]
        return x, x_short, x_diff, skip, skip_short, skip_diff

    def fun2(self, x, x_short, x_diff, skip, skip_short, skip_diff):
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if self.use_RevIN:
            x = self.revin(x.transpose(2, 3), 'denorm').transpose(2, 3)
        return x

    def fun3(self, x):
        new_supports = None
        gl_loss = None
        x_short, x_diff = None, None
        dynamic_adj_pos, dynamic_adj_neg = None, None
        if self.gcn_bool:
            adp = self.graph_construct.get_static()
            # gl_loss = gl_loss_
            if self.addaptadj:
                new_supports = adp
            if self.short_bool:
                x_long = torch.einsum("bckj,ik->bcij", x, torch.matrix_power(adp, self.smooth_depth))
                x_short = x - x_long
                x_diff = x[..., 1:] - x[..., :-1]
                x_diff = F.pad(x_diff, (1, 0))

                dynamic_adj_pos, dynamic_adj_neg = self.graph_construct.get_dynamic(x_short[..., -self.seq_length:],
                                                                                    x_diff[..., -self.seq_length:])

        return x, x_short, x_diff, new_supports, dynamic_adj_pos, dynamic_adj_neg

    def forward(self, input, pred_time_embed=None, MLP_hidden=None):
        if self.use_RevIN:
            if self.checkpoint:
                input = checkpoint(self.revin, input.permute(0, 3, 1, 2), 'norm').permute(0, 2, 3, 1)
            else:
                input = self.revin(input.permute(0, 3, 1, 2), 'norm').permute(0, 2, 3, 1)

        in_len = input.size(3)
        x = input

        if self.checkpoint:
            x, x_short, x_diff, new_supports, dynamic_adj_pos, dynamic_adj_neg = checkpoint(self.fun3, x)
        else:
            x, x_short, x_diff, new_supports, dynamic_adj_pos, dynamic_adj_neg = self.fun3(x)
        x_short, x_diff = None, None
        skip_short, skip_diff = None, None
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv1(x)
        x = self.fusion_start_mlp(x,MLP_hidden)
        if self.patch_num < self.receptive_field:
            x = nn.functional.pad(x, [(self.receptive_field - self.patch_num), 0, 0, 0])

        # WaveNet layers
        for i in range(self.layers):
            if self.checkpoint:
                x, x_short, x_diff, skip, skip_short, skip_diff = checkpoint(self.fun1, i, x, x_short, x_diff, skip,
                                                                             skip_short, skip_diff, new_supports,
                                                                             dynamic_adj_pos, dynamic_adj_neg,
                                                                             MLP_hidden)
            else:
                x, x_short, x_diff, skip, skip_short, skip_diff = self.fun1(i, x, x_short, x_diff, skip,
                                                                            skip_short, skip_diff, new_supports,
                                                                            dynamic_adj_pos, dynamic_adj_neg,
                                                                            MLP_hidden)

        if self.checkpoint:
            x = checkpoint(self.fun2, x, x_short, x_diff, skip, skip_short, skip_diff)
        else:
            x = self.fun2(x, x_short, x_diff, skip, skip_short, skip_diff)

        return x, None, None


class gwnet(nn.Module):

    def __init__(self, device, short_bool, smooth_depth, args, distribute_data=None):
        super(gwnet, self).__init__()
        self.device = device
        self.short_bool = short_bool
        self.smooth_depth = smooth_depth
        num_nodes = args.num_nodes
        dropout = args.dropout
        self.checkpoint = args.checkpoint

        in_dim = args.in_dim
        residual_channels = args.residual_channels
        # dilation_channels = args.nhid
        skip_channels = args.skip_channels
        end_channels = args.end_channels

        layers = args.layers

        self.dropout = dropout

        self.layers = args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj

        self.nodes = num_nodes
        self.seq_length = args.lag
        self.patch_len = args.patch_len
        self.patch_num = self.seq_length // self.patch_len

        kernel_size = 7  # 12 #

        self.part1 = gwnet_part1(self.device, self.short_bool, 1, args)

        self.MLP_component = nn.Sequential(STID(num_nodes=self.nodes, input_len=args.lag, output_len=args.output_len,
                                                num_layer=args.MLP_layer, input_dim=args.MLP_indim,
                                                node_dim=args.MLP_dim,
                                                embed_dim=args.MLP_dim,
                                                temp_dim_tid=args.MLP_dim, temp_dim_diw=args.MLP_dim,
                                                if_T_i_D=args.if_T_i_D,
                                                if_D_i_W=args.if_D_i_W, if_node=args.if_node, first_time=args.s_period,
                                                second_time=args.b_period, time_norm=args.time_norm,
                                                patch_len=self.patch_len),
                                           nn.Conv2d(in_channels=args.MLP_dim * 4, out_channels=residual_channels,
                                                     kernel_size=(1, 1),
                                                     bias=True))

    def forward(self, input, pred_time_embed=None):
        MLP_hidden = self.MLP_component(pred_time_embed)
        # MLP_part = MLP_hidden  ##################是否加入detach####################
        # MLP_hidden = self.mlp_(MLP_hidden)
        # MLP_hidden = None

        x, _, _ = self.part1(input, None, MLP_hidden)

        return x, None, None
