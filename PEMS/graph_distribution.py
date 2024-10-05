import torch
from torch import nn
from layer_distribution_module import *
from torch.nn.utils import weight_norm
from layer_distribution_mtgnn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)

    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y, y_soft



class graph_constructor(nn.Module):
    def __init__(self, in_dim, num_nodes, embed_dim, device, patch_len=12):
        super(graph_constructor, self).__init__()
        self.nodes = num_nodes
        self.emb_dim = embed_dim
        self.patch_len = patch_len

        self.embed1 = nn.Embedding(self.nodes, embed_dim)
        self.device = device

        # self.embed1 = nn.Embedding(nodes, dim)  # 静态图节点嵌入
        self.norm_static = nn.LayerNorm(self.nodes)

        self.short_conv = nn.Conv2d(in_dim, 16, (1, 5), padding='same')
        self.short_line = nn.Parameter(torch.randn(16, patch_len, self.emb_dim), requires_grad=True)
        self.diff_conv = nn.Conv2d(in_dim, 16, (1, 5), padding='same')
        self.diff_line = nn.Parameter(torch.randn(16, patch_len, self.emb_dim), requires_grad=True)
        self.long_line = nn.Linear(self.emb_dim, self.emb_dim)
        self.norm_dynamic1 = nn.LayerNorm(self.emb_dim)

        self.diag = + torch.eye(self.nodes, device=self.device, dtype=torch.float) * 1e-5



    def get_static(self):
        idx = torch.arange(self.nodes).to(self.device)
        nodevec = self.embed1(idx)
        th = torch.tensor(1 / self.nodes, device=self.device, dtype=torch.float)

        adj = F.relu(torch.mm(nodevec, nodevec.transpose(0, 1)))
        adj = adj / torch.sum(adj, dim=-1, keepdim=True)
        return adj

    def get_dynamic(self, input, input_diff):
        # input,input_diff B×C×N×T
        B, C, N, T = input.shape
        patch_num = T // self.patch_len
        if patch_num * self.patch_len != T:
            raise ValueError('graph_constructor time length')

        idx = torch.arange(self.nodes).to(self.device)
        nodevec_static = self.embed1(idx)
        nodevec_static = self.long_line(nodevec_static)

        x = self.short_conv(input)
        C = x.shape[1]
        x = x.view(B, C, N, patch_num, -1)  # B×C×N×patch_num×patch_len
        x_diff = self.diff_conv(input_diff)
        x_diff = x_diff.view(B, C, N, patch_num, -1)

        x = torch.einsum("bcnpl,cld->bpnd", x, self.short_line)
        x_diff = torch.einsum("bcnpl,cld->bpnd", x_diff, self.diff_line)

        x_patch = x + torch.sigmoid(x_diff) * nodevec_static
        x_patch = self.norm_dynamic1(x_patch)
        dynamic_adj = torch.einsum("bpik,bpjk->bpij", x_patch, x_patch) #/ torch.sqrt(torch.tensor(self.emb_dim, device=input.device))
        # dynamic_adj =
        # 正边
        dy_pos = F.relu(dynamic_adj) + self.diag
        dy_pos = dy_pos / torch.sum(dy_pos,dim=-1,keepdim=True)
        #
        # # 负边
        # dy_neg = F.relu(-1 * dynamic_adj) + self.diag
        # dy_neg = dy_neg / torch.sum(dy_neg,dim=-1,keepdim=True)
        # dy_neg = dy_neg*(1-torch.eye(self.nodes,device=self.device))
        # dy_neg = -1 * dy_neg

        return dy_pos
