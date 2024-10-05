import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_distribution import graph_constructor
from layer_distribution_mtgnn import *
from STID_arch import STID
from layer_distribution_module import RevIN


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.einsum('nvw, ncvl->ncwl', [A, x])
        x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        return x.contiguous()


class patch_nconv(torch.nn.Module):
    def __init__(self):
        super(patch_nconv, self).__init__()

    def forward(self, x, A):
        # x:B×C×N×L×P
        # A:B×P×N×N
        # return B×C×N×L×P
        x = torch.einsum("bcwlp,bpvw->bcvlp", (x, A))
        return x.contiguous()


class gcn_patch_model(torch.nn.Module):
    def __init__(self, c_in, c_out, gcn_depth, seq_length, patch_len, dropout, alpha=0.02):
        super(gcn_patch_model, self).__init__()
        self.patch_nconv = patch_nconv()
        self.gcn_depth = gcn_depth
        self.seq_length = seq_length
        self.patch_len = patch_len
        self.dropout = dropout
        # self.theta = torch.nn.Parameter(torch.ones((gcn_depth, 2), dtype=torch.float) * alpha, requires_grad=True)
        self.mlp = torch.nn.Conv2d((1 + gcn_depth) * c_in, c_out, kernel_size=(1, 1))

        self.mlp_rest = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # 当输入序列的长度大于seq_length，对前面多出来的部分进行通道变换

    def forward(self, A, x):
        # x:B×C×N×T
        # A:B×P×N×N
        B, C, N, T = x.shape
        patch_num = T // self.patch_len
        out = []
        if T > self.seq_length:
            # x_patch = torch.split(x[..., -self.seq_length:], self.patch_len, dim=3)
            # x_patch = torch.stack(x_patch,dim=-1)

            x_patch = x[..., -self.seq_length:].view(B, C, N, self.seq_length // self.patch_len, -1).transpose(-1,
                                                                                                               -2).contiguous()
            out.append(x[..., -self.seq_length:])
        else:
            if patch_num * self.patch_len == T:
                A = A[:, -patch_num:, :, :]
                x_patch = x.view(B, C, N, patch_num, -1).transpose(-1, -2).contiguous()
                out.append(x)
            else:
                A = A[:, -(patch_num + 1):, :, :]
                x_temp = torch.nn.functional.pad(x, ((patch_num + 1) * self.patch_len - T, 0, 0, 0, 0, 0, 0, 0))
                x_patch = x_temp.view(B, C, N, patch_num + 1, -1).transpose(-1, -2).contiguous()
                out.append(x_temp)

        h = x_patch
        for i in range(self.gcn_depth):
            h = self.patch_nconv(h, A)
            out.append(h.transpose(-1, -2).contiguous().view(*h.shape[:3], -1).contiguous())

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        if T > self.seq_length:
            h_rest = self.mlp_rest(x[..., :T - self.seq_length])
            ho = torch.cat([h_rest, ho], dim=-1)
        elif patch_num * self.patch_len < T:
            ho = ho[..., -T:]
        return F.dropout(ho, p=self.dropout, training=self.training)


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_module, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, AL, AS_pos, short_bool=True, smooth_depth=1):
        # AL长期分量的邻接矩阵
        # AS短期分量的邻接矩阵
        out = [x]
        if short_bool:
            x_long = torch.einsum("bckj,ik->bcij", x, torch.matrix_power(AL, smooth_depth))
            x_short = x - x_long

        for k in range(1, self.order + 1):
            if short_bool:
                x_long = self.nconv(x_long, AL)
                x_short = self.nconv(x_short, AS_pos)
                out.append(x_long + x_short)
            else:
                x = self.nconv(x, AL)
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
        mlp_part = mlp_part.expand(-1, -1, -1, T)
        rate = torch.cat([x, mlp_part], dim=1)
        rate = torch.sigmoid(self.conv(rate))
        out = rate * x + (1 - rate) * mlp_part
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


class SDGL_SeqDe_part1(nn.Module):
    def __init__(self, short_bool, smooth_depth, args, distribute_data=None):
        super(SDGL_SeqDe_part1, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.short_bool = short_bool
        num_nodes = args.num_nodes
        dropout = args.dropout

        in_dim = args.in_dim
        residual_channels = args.nhid
        dilation_channels = args.nhid
        skip_channels = args.nhid * 8
        end_channels = args.nhid * 16

        self.dropout = dropout
        self.layers = args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj
        self.smooth_depth = smooth_depth

        # self.filter_convs = nn.ModuleList()
        # self.gate_convs = nn.ModuleList()
        self.time_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fusion_t_mlp = nn.ModuleList()
        self.fusion_g_mlp = nn.ModuleList()
        self.norm = nn.ModuleList()

        if self.short_bool:
            # self.filter_convs_short = nn.ModuleList()
            # self.gate_convs_short = nn.ModuleList()
            self.time_convs_short = nn.ModuleList()
            self.residual_convs_short = nn.ModuleList()
            self.skip_convs_short = nn.ModuleList()
            self.bn_short = nn.ModuleList()
            self.gconv_short = nn.ModuleList()
            self.norm_short = nn.ModuleList()

            # self.filter_convs_diff = nn.ModuleList()
            # self.gate_convs_diff = nn.ModuleList()
            self.time_convs_diff = nn.ModuleList()
            self.residual_convs_diff = nn.ModuleList()
            self.skip_convs_diff = nn.ModuleList()
            self.bn_diff = nn.ModuleList()
            self.gconv_diff = nn.ModuleList()
            self.norm_diff = nn.ModuleList()
            self.fusion_t = nn.ModuleList()
            self.fusion_g = nn.ModuleList()

        self.nodes = num_nodes

        # 时间窗口
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        if self.short_bool:
            self.start_conv_short = nn.Conv2d(in_channels=in_dim,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
            self.start_conv_diff = nn.Conv2d(in_channels=in_dim,
                                             out_channels=residual_channels,
                                             kernel_size=(1, 1))
        self.seq_length = args.lag
        kernel_size = 7

        dilation_exponential = args.dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** self.layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        rf_size_i = 1
        new_dilation = 1
        num = 1

        for j in range(1, self.layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.time_convs.append(Tiem_conv(num * residual_channels, dilation_channels, dilation_factor=new_dilation))

            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
            self.fusion_t_mlp.append(Fusioncell(dilation_channels))
            self.fusion_g_mlp.append(Fusioncell(residual_channels))
            if self.short_bool:
                self.time_convs_short.append(
                    Tiem_conv(num * residual_channels, dilation_channels, dilation_factor=new_dilation))

                self.residual_convs_short.append(nn.Conv2d(in_channels=dilation_channels,
                                                           out_channels=residual_channels,
                                                           kernel_size=(1, 1)))

                self.time_convs_diff.append(
                    Tiem_conv(num * residual_channels, dilation_channels, dilation_factor=new_dilation))
                self.fusion_t.append(Fusionshort(dilation_channels))

                self.residual_convs_diff.append(nn.Conv2d(in_channels=dilation_channels,
                                                          out_channels=residual_channels,
                                                          kernel_size=(1, 1)))

            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
                if self.short_bool:
                    self.skip_convs_short.append(nn.Conv2d(in_channels=dilation_channels,
                                                           out_channels=skip_channels,
                                                           kernel_size=(1, self.seq_length - rf_size_j + 1)))
                    self.skip_convs_diff.append(nn.Conv2d(in_channels=dilation_channels,
                                                          out_channels=skip_channels,
                                                          kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                if self.short_bool:
                    self.skip_convs_short.append(nn.Conv2d(in_channels=dilation_channels,
                                                           out_channels=skip_channels,
                                                           kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                    self.skip_convs_diff.append(nn.Conv2d(in_channels=dilation_channels,
                                                          out_channels=skip_channels,
                                                          kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            if self.gcn_bool:
                self.gconv.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))
                if self.short_bool:
                    self.gconv_short.append(
                        gcn_patch_model(residual_channels, residual_channels, 2, seq_length=self.seq_length,
                                        patch_len=12, dropout=self.dropout))
                    self.gconv_diff.append(
                        gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))
                    self.fusion_g.append(Fusionshort(residual_channels))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                           elementwise_affine=True))
                if self.short_bool:
                    self.norm_short.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                                     elementwise_affine=True))
                    self.norm_diff.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                                    elementwise_affine=True))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                           elementwise_affine=True))
                if self.short_bool:
                    self.norm_short.append(
                        LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=True))
                    self.norm_diff.append(
                        LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=True))
            new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=args.lag,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)
        self.idx = torch.arange(self.nodes).to(device)

        # K = (self.nodes * args.rate_K)
        self.graph_construct = graph_constructor(args.in_dim, self.nodes, args.embed_dim, device, patch_len=12)

        # self.MLP_component = STID(num_nodes=self.nodes, input_len=args.horizon, output_len=args.lag,
        #                           num_layer=args.MLP_layer, input_dim=args.MLP_indim, node_dim=args.MLP_dim,
        #                           embed_dim=args.MLP_dim,
        #                           temp_dim_tid=args.MLP_dim, temp_dim_diw=args.MLP_dim, if_T_i_D=args.if_T_i_D,
        #                           if_D_i_W=args.if_D_i_W, if_node=args.if_node, first_time=args.s_period,
        #                           second_time=args.b_period, time_norm=args.time_norm)
        #
        # self.mlp_ = nn.Conv2d(in_channels=128, out_channels=residual_channels, kernel_size=(1, 1), bias=True)

        self.use_RevIN = args.use_RevIN
        if args.use_RevIN:
            self.revin = RevIN(args.num_nodes)
            # self.revin_short = RevIN(args.num_nodes)
            # self.revin_diff = RevIN(args.num_nodes)
            # self.revin_mlp = RevIN(args.num_nodes)

    def forward(self, input, pred_time_embed=None, mlp_component=None):
        if self.use_RevIN:
            input = self.revin(input.permute(0, 3, 1, 2), 'norm').permute(0, 2, 3, 1)

        # MLP_out, MLP_hidden = self.MLP_component(pred_time_embed)
        # MLP_part = MLP_hidden
        # # Features extracted from the identification module
        # mlp_component = self.mlp_(MLP_part)

        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = input

        new_supports = None
        gl_loss = None
        dynamic_adj_pos = None

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

                dynamic_adj_pos = self.graph_construct.get_dynamic(x_short[..., -self.seq_length:],
                                                                   x_diff[..., -self.seq_length:])
                dynamic_adj_pos = torch.mean(dynamic_adj_pos, dim=1)
                # dynamic_adj_neg = torch.mean(dynamic_adj_neg,dim=1)

        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        # if self.short_bool:
        #     x_short = self.start_conv(x_short)
        #     x_diff = self.start_conv_diff(x_diff)
        # WaveNet layers
        for i in range(self.layers):
            residual = x
            # if self.short_bool:
            #     residual_short = x_short
            #     residual_diff = x_diff
            # dilated convolution
            x = self.time_convs[i](x)

            # if self.short_bool:
            #     x_short = self.time_convs[i](x_short)
            #     x_diff = self.time_convs_diff[i](x_diff)

            # x = self.fusion_t[i](x, x_diff, x_short)
            x = self.fusion_t_mlp[i](x, mlp_component)

            x = F.dropout(x, self.dropout, training=self.training)
            # if self.short_bool:
            #     x_short = F.dropout(x_short, self.dropout, training=self.training)
            #     x_diff = F.dropout(x_diff, self.dropout, training=self.training)

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_bool:
                x = self.gconv[i](x, new_supports, dynamic_adj_pos, self.short_bool, self.smooth_depth)
                # if self.short_bool:
                #     # x_short = self.gconv_short[i](dynamic_adj, x_short)
                #     x_short = self.gconv[i](x_short,dynamic_adj_pos) + self.gconv[i](x_short,dynamic_adj_neg)
                #     x_diff = self.gconv_diff[i](x_diff, new_supports)
            else:
                x = self.residual_convs[i](x)
                # if self.short_bool:
                #     x_short = self.residual_convs[i](x_short)
                #     x_diff = self.residual_convs_diff[i](x_diff)

            # if self.short_bool:
            #     x = self.fusion_g[i](x, x_diff, x_short)
            x = self.fusion_g_mlp[i](x, mlp_component)

            x = x + residual[:, :, :, -x.size(3):]
            # x = self.norm[i](x, self.idx)

            # if self.short_bool:
            #     x_short = x_short + residual_short[:, :, :, -x_short.size(3):]
            #     # x_short = self.norm_short[i](x_short, self.idx)
            #
            #     x_diff = x_diff + residual_diff[:, :, :, -x_diff.size(3):]
            #     x_diff = self.norm_diff[i](x_diff, self.idx)

        skip = self.skipE(x) + skip

        x = F.relu(skip)  # + mlp_skip
        x = F.relu(self.end_conv_1(x))

        x = self.end_conv_2(x)

        if self.use_RevIN:
            x = self.revin(x.transpose(2, 3), 'denorm').transpose(2, 3)

        return x, gl_loss, None


class SDGL_SeqDe(nn.Module):
    def __init__(self, short_bool, smooth_depth, args, distribute_data=None):
        super(SDGL_SeqDe, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.short_bool = short_bool
        num_nodes = args.num_nodes
        dropout = args.dropout

        in_dim = args.in_dim
        residual_channels = args.nhid
        dilation_channels = args.nhid
        skip_channels = args.nhid * 8
        end_channels = args.nhid * 16

        self.dropout = dropout
        self.layers = args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj
        self.smooth_depth = smooth_depth

        self.nodes = num_nodes

        self.seq_length = args.lag
        kernel_size = 7

        self.idx = torch.arange(self.nodes).to(device)

        self.part1 = SDGL_SeqDe_part1(self.short_bool, self.smooth_depth, args)
        hide = (int(args.if_node) + int(args.if_T_i_D) + int(args.if_D_i_W) + 1) * args.MLP_dim
        self.MLP_component = nn.Sequential(STID(num_nodes=self.nodes, input_len=args.horizon, output_len=args.lag,
                                                num_layer=args.MLP_layer, input_dim=args.MLP_indim,
                                                node_dim=args.MLP_dim,
                                                embed_dim=args.MLP_dim,
                                                temp_dim_tid=args.MLP_dim, temp_dim_diw=args.MLP_dim,
                                                if_T_i_D=args.if_T_i_D,
                                                if_D_i_W=args.if_D_i_W, if_node=args.if_node, first_time=args.s_period,
                                                second_time=args.b_period, time_norm=args.time_norm),
                                           nn.Conv2d(in_channels=hide, out_channels=residual_channels,
                                                     kernel_size=(1, 1), bias=True))

    def forward(self, input, pred_time_embed=None):
        mlp_component = self.MLP_component(pred_time_embed)

        x, _, _ = self.part1(input, None, mlp_component)

        return x, None, None
