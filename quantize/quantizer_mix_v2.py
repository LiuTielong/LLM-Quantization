"""
这个混合bit量化，从外界文件接收bit位，而不是bit位可学。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
from utils import NativeScalerWithGradNormCount

CLIPMIN = 1e-5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def floor_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x

def ceil_ste(x: torch.Tensor):
    return (x.ceil() - x).detach() + x

class Mix2AffineQuantizer(nn.Module):
    def __init__(
        self,
        target_bit: int=4,                      # 目标bit位                              
        symmetric: bool = False,
        n_bits: torch.Tensor=[], 
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_channel",
        group_size=None,
        shape=None,
        mix2_nbits:torch.Tensor=[],
        lwc=False
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric

        self.target_bit = target_bit
        self.n_bits = mix2_nbits.unsqueeze(dim=-1)
        self.q_min = torch.zeros_like(self.n_bits)
        self.q_max = 2 ** (self.n_bits) - 1

        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.mode = "calibration"
        
        init_value = 4.             # init value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size


    def fake_quant(self, x, scale, round_zero_point, qmin, qmax):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(qmin, qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        self.shape = x.shape
        n_bits = self.n_bits
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:                                                                                                    # lwc只有权重量化才有
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (n_bits - 1) - 1)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(scale)
        else:
            range = xmax - xmin
            scale = range / (2 ** self.n_bits - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = - xmin / scale
        round_zero_point = zero_point.clamp(min=-1e4,max=1e4).round()

        x_dequant = self.fake_quant(x, scale, round_zero_point, self.q_min, self.q_max)

        return x_dequant

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)                                                                      # 注册到buffer里面，之后移动device的时候就很方便了
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def free(self):
        self.cached_xmax = None
        self.cached_xmin = None

    """
    本来是用来计算模型的size惩罚项的，后面我为了多个layer的惩罚项一起计算，就
    让这个函数返回目标size和实际size。
    """
    """
    将所有的bit位都round成整数
    """
    def round_nbits(self):
        self.n_bits_round = self.n_bits.data
        self.n_bits_round = round_ste(self.n_bits_round + self.threshold)
        del self.n_bits
        self.n_bits = self.n_bits_round




def main():
    torch.manual_seed(10)
    x = torch.rand([4, 10000]).to("cuda")                                                                                             # 对x来说，input_channel是列，也就是10列。而output channel有4行。
    # n_bits = torch.tensor([[6.8],[7.5],[6.2], [5.7]])
    n_bits = torch.tensor([6,6,6,6]).to("cuda")                                                                                            # 我取一个非整数的target_bit，就迫使n_bits的搜索既有3又有4，如果我取target_bit=3, 就可能最后bit位都是3了
    quantizer = Mix2AffineQuantizer(mix2_nbits=n_bits, shape=x.shape)
    k = 0.01

    x_dequant = quantizer(x)
    print(torch.mean(torch.abs(x-x_dequant)))

if __name__=="__main__":
    main()