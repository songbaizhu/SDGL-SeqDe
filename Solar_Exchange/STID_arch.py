import torch
from torch import nn

from mlp import MultiLayerPerceptron


class STID(nn.Module):

    def __init__(self, num_nodes, input_len, output_len, num_layer, input_dim=3, node_dim=32, embed_dim=32,
                 temp_dim_tid=32, temp_dim_diw=32, if_T_i_D=True, if_D_i_W=True, if_node=True, first_time=288,
                 second_time=7, time_norm=True):
        super().__init__()
        # attributes
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw

        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.if_spatial = if_node
        self.first_time = first_time
        self.second_time = second_time
        self.time_norm = time_norm

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Embedding(self.first_time, self.temp_dim_tid) # 288
            nn.init.xavier_uniform_(self.time_in_day_emb.weight)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Embedding(self.second_time, self.temp_dim_diw) # 7
            nn.init.xavier_uniform_(self.day_in_week_emb.weight)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        # self.regression_layer = nn.Conv2d(
        #     in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, **kwargs):
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            if self.time_norm:
                time_in_day_emb = self.time_in_day_emb((
                    t_i_d_data[:, -1, :] * self.first_time).type(torch.LongTensor).to(input_data.device))
            else:
                time_in_day_emb = self.time_in_day_emb((t_i_d_data[:, -1, :]).type(torch.LongTensor).to(input_data.device))

        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb((
                d_i_w_data[:, -1, :]).type(torch.LongTensor).to(input_data.device))
        else:
            day_in_week_emb = None

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)

        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        hidden = self.encoder(hidden)
        # prediction = self.regression_layer(hidden)
        return None, hidden
