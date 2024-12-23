import os
import math
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from pytorch_msssim import ssim
from models.FGM import FGM
from math import sqrt
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
# import clip
from tqdm import tqdm
import pyiqa
# load clip
# c_model, preprocess = clip.load( "/home/ubuntu/Image-restoration/CycleRDM/CLIP/ViT-B-32.pt", device=torch.device("cpu"))  # ViT-B/32
# c_model.to(device)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class FrequencyTransform(nn.Module):
    def __init__(self):
        super(FrequencyTransform, self).__init__()

    def forward(self, dp):
        dp = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)
        return dp_amp, dp_pha

class WaveletTransform(nn.Module):
    def __init__(self):
        super(WaveletTransform, self).__init__()
        self.requires_grad = False

    @staticmethod
    def dwt(x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

    @staticmethod
    def iwt(x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch = int(in_batch / (r**2))
        out_channel, out_height, out_width = in_channel, r * in_height, r * in_width
        x1 = x[0:out_batch, :, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    def forward(self, x, inverse=False):
        if inverse:
            return self.iwt(x)
        else:
            return self.dwt(x)

class Normalize:
    @staticmethod
    def apply(x):
        ymax = 255
        ymin = 0
        xmax = x.max()
        xmin = x.min()
        return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                 num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps,
                                  1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device


        self.high_enhance0 = FGM(in_channels=3, out_channels=64)

        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b,dm_num=True, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        skip_1 = self.config.diffusion.num_diffusion_timesteps_1 // self.args.sampling_timesteps
        seq_1 = range(0, self.config.diffusion.num_diffusion_timesteps_1, skip_1)



        n, c, h, w = x_cond.shape

        seq_next = [-1] + list(seq[:-1])
        seq_next_1 = [-1] + list(seq[:-1])

        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        # for i, j in zip(reversed(seq), reversed(seq_next)):
        for i, j in zip(reversed(seq), reversed(seq_next)) if dm_num else zip(reversed(seq_1), reversed(seq_next_1)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        # return xs[-1]
        return xs

    def forward(self, x):
        data_dict = {}
        dwt, idwt = WaveletTransform(), WaveletTransform()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]


        b = self.betas.to(input_img.device)


        b1 = self.betas.to(input_LL.device)
        t1 = torch.randint(low=0, high=self.num_timesteps, size=(
             input_LL.shape[0] // 2 + 1,)).to(self.device)
        t1 = torch.cat([t1, self.num_timesteps - t1 - 1],
                      dim=0)[: input_LL.shape[0]].to(x.device)
        a1 = (1 - b1).cumprod(dim=0).index_select(0, t1).view(-1, 1, 1, 1)
        e1 = torch.randn_like(input_LL)



        b2 = self.betas.to(input_img.device)
        t2 = torch.randint(low=0, high=self.num_timesteps, size=(
             input_img.shape[0] // 2 + 1,)).to(self.device)
        t2 = torch.cat([t2, self.num_timesteps - t2 - 1],
                      dim=0)[: input_img.shape[0]].to(x.device)
        a2 = (1 - b2).cumprod(dim=0).index_select(0, t2).view(-1, 1, 1, 1)
        e2 = torch.randn_like(input_img)



        if self.training==False:
            img_list = self.sample_training(input_img, b)
            pred_x= img_list[-1]

            pred_x_list_1 = self.sample_training(pred_x, b2)
            pred_x_1=pred_x_list_1[-1]

            pred_x_dwt = dwt.dwt(pred_x_1)
            pred_x_LL, pred_x_high0 = pred_x_dwt[:n, ...], pred_x_dwt[n:, ...]
            pred_LL_list = self.sample_training(pred_x_LL, b1)
            pred_LL = pred_LL_list[-1]
            pred_x_high0 = self.high_enhance0(pred_x_high0)
            pred_x_2 = idwt.iwt(torch.cat((pred_LL, pred_x_high0), dim=0))

            # data_dict["pred_x"] = pred_x
            # data_dict["pred_x_1"] = pred_x_1
            data_dict["pred_x_2"] = pred_x_2

        return data_dict



class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        # self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        # self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='rgb')
        self.model = Net(args, config) 
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)  

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("Load checkpoint: ", os.path.exists(load_path))
        print("Current checkpoint: {}".format(load_path))

    
    
