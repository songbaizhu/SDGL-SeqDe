import torch.optim as optim
from model_distribution import *
from util import *
from torch.optim import lr_scheduler
import torch


class trainer():
    def __init__(self, device, data, distribute_data, args):

        self.device = device
        self.model = gwnet(device=device, short_bool=args.short_bool, smooth_depth=args.smooth_depth, args=args,
                           distribute_data=distribute_data)

        self.model.to(device)
        # self.gc_order = args.order

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                        patience=10, eps=0.00001, cooldown=20, verbose=True)

        self.loss = nn.L1Loss(size_average=False).to(device)
        self.loss_mse = nn.MSELoss(size_average=False).to(device)

        nparams = sum([p.nelement() for p in self.model.parameters()])
        print(nparams)
        self.clip = 0.5
        self.loss_usual = nn.SmoothL1Loss()
        self.accumulation_steps = args.accumulation_steps

    def train(self, input, real_val, data, pred_time_embed=None, iter=0):
        accumulation_steps = self.accumulation_steps
        self.model.train()
        output, gl_loss, _ = self.model(input, pred_time_embed)
        output = output.squeeze()
        real_val = real_val.squeeze()

        real = real_val
        scale = data.scale  # .expand(real.size(0), data.m)
        real, predict = real * scale, output * scale

        if gl_loss is None:
            loss = self.loss(predict, real)
        else:
            loss = self.loss(predict, real) + torch.mean(gl_loss)
        loss = loss / accumulation_steps
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        if ((iter + 1) % accumulation_steps) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def eval(self, input, real_val, data, pred_time_embed=None):
        self.model.eval()
        with torch.no_grad():
            output, _, _, = self.model(input, pred_time_embed)
            output = output[:, -1, :, :].clone()
        output = output.squeeze(dim=-1)
        # if output.dim() < 2:
        #     output = output.unsqueeze(dim=0)
        real = real_val

        scale = data.scale  # .expand(real.size(0), data.m)
        real, predict = real * scale, output * scale
        loss_mse = self.loss_mse(predict, real)
        loss = self.loss(predict, real)
        samples = (output.size(0) * data.m)
        return loss.item(), loss_mse.item(), samples, output
