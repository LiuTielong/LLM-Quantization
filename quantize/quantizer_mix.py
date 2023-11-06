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

"""
仅仅针对权重作混合bit量化。
对于权重的bit位的搜索,有两个阶段。
第一阶段是bit位都用实数表示,第2阶段则用整数。
fake_quant函数基本不变。输入仍然是x, scale, round_zero_point。
forward()函数就改成只能支持权重的混合bit量化. 计算公式如论文Fracbits。
考虑到我的权重后面要支持gptq, fake_quant函数不变,而是在per_token_dynamic_calibration函数中两次调用fake_quant()函数.
在forward()函数中就不调用fake_quant()函数了。(或者不要calibration了,直接都写在forward里面)。
"""


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def floor_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x

def ceil_ste(x: torch.Tensor):
    return (x.ceil() - x).detach() + x

class MixAffineQuantizer(nn.Module):
    def __init__(
        self,
        target_bit: int=4,                      # 目标bit位
        n_bits: torch.Tensor=[],                
        hard_assign=False,                      # 初始是false，改为true的时候表示所有的bit位都要改成int进行计算了。     
        threshold=0.0,                          # 仿照Fracbits里面加入了一个threshold                  
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_channel",
        group_size=None,
        shape=None,
        lwc=False
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric

        self.target_bit = target_bit
        # self.n_bits = n_bits
        if n_bits == []:
            len = shape[0]
            n_bits = (torch.ones(len,1) * self.target_bit + 0.5 + torch.rand(len, 1) / 4).to("cuda")
        self.n_bits = nn.Parameter(n_bits)              # 创建参数
        self.hard_assign = hard_assign
        self.threshold=threshold

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

        """
        A example. opt-125m, layer 0, k_proj, weight_quantizer.
        Quantize weight to 3 bits, qmin = 0, qmax = 7. 
        self.metric = "minmax". 
        self.dynamic_method = per_channel.
        lwc = True, group_size = None.
        self.unbound_factor = self.lowbound_factor = 4 for shape[768,1]
        """

        """
        A example. opt-125m, w3a16, layer 0, qkt_quant, x1_quantizer.
        symmetric=false, dynamic=false, dynamic_method=per_token

        """


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
        self.n_bits.data = self.n_bits.data.clamp(2, 8)                                                                         # 非常重要，否则量化过程中会导致x变成nan
        n_bits = self.n_bits
        if self.hard_assign:
            n_bits = round_ste(n_bits + self.threshold)                                                                        # 在第二阶段，bit数要换成整数                                                                                           
        assert torch.all(n_bits >= 2), "有量化bit位太小了"
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
        n_bits_low = floor_ste(n_bits)
        n_bits_high = ceil_ste(n_bits)                                                                                  # 如何n_bits为整数，那么low=high
        qmin_low = torch.zeros_like(n_bits_low)
        qmax_low = 2 ** (n_bits_low) - 1
        qmin_high = torch.zeros_like(n_bits_high)
        qmax_high = 2 ** n_bits_high - 1
        high_prop = n_bits - n_bits_low
        low_prop = 1.0 - high_prop
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale_low = abs_max / (2 ** (n_bits_low - 1) - 1)
            scale_high = abs_max / (2 ** (n_bits_high - 1) - 1)
            zero_point_low = (2 ** (n_bits_low - 1) - 1)*torch.ones_like(scale_low)   
            zero_point_high = (2 ** (n_bits_high - 1) - 1)*torch.ones_like(scale_high)
        else:
            range = xmax - xmin
            scale_low = range / (2 ** n_bits_low - 1)
            scale_high = range / (2 ** n_bits_high - 1)
            scale_low = scale_low.clamp(min=CLIPMIN, max=1e4)
            scale_high = scale_high.clamp(min=CLIPMIN, max=1e4)
            zero_point_low = -xmin / scale_low
            zero_point_high = -xmin / scale_high
        round_zero_point_low = zero_point_low.clamp(min=-1e4,max=1e4).round()
        round_zero_point_high = zero_point_high.clamp(min=-1e4,max=1e4).round()

        x_dequant = low_prop * self.fake_quant(x, scale_low, round_zero_point_low, qmin_low, qmax_low) \
                    + high_prop * self.fake_quant(x, scale_high, round_zero_point_high, qmin_high, qmax_high)

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
    def model_size_loss(self):
        loss_func = torch.nn.MSELoss()
        prop = self.shape[0] * self.shape[1]                                                                            # 除以这个权重矩阵的大小，得到的loss就是平均每个权重的损失
        n_bits = self.n_bits
        if self.hard_assign:
            n_bits = round_ste(n_bits + self.threshold)
        self.target_size = self.shape[0] * self.shape[1] * self.target_bit
        self.target_size = torch.tensor(self.target_size).to("cuda") / prop
        self.model_size = torch.sum(n_bits * self.shape[1]) / prop
        # loss = torch.abs(self.model_size - self.target_size) / prop
        # loss = loss_func(self.model_size, self.target_size)
        return self.model_size, self.target_size

    def set_hard(self):
        self.hard_assign = True

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
    x[1] = x[1] * 100                                                                                                   # 制造出非常难以量化的某一行
    # n_bits = torch.tensor([[6.8],[7.5],[6.2], [5.7]])
    n_bits = torch.tensor([[4.8],[4.5],[4.2], [4.7]]).to("cuda")
    target_bit = 3.5                                                                                                    # 我取一个非整数的target_bit，就迫使n_bits的搜索既有3又有4，如果我取target_bit=3, 就可能最后bit位都是3了
    quantizer = MixAffineQuantizer(target_bit=target_bit, n_bits=[], shape=x.shape)
    k = 0.01
    # 创建一个优化器
    optimizer = torch.optim.AdamW(
        [{"params":quantizer.n_bits, "lr":0.1}]
    )
    loss_scaler = NativeScalerWithGradNormCount()
    mse_loss = torch.nn.MSELoss()

    for epoch in range(75):  # 第一组epochs是优化连续的nbits
        x_dequant = quantizer(x)
        loss_quant = mse_loss(x, x_dequant)
        model_size, target_size = quantizer.model_size_loss()
        loss_size = mse_loss(model_size, target_size)
        loss = loss_size * k + loss_quant

        print(loss_size)
        print(loss_quant)
        print(quantizer.n_bits)
        optimizer.zero_grad()
        norm = loss_scaler(loss, optimizer, parameters=quantizer.n_bits)
        

    print("---------------------------------------------------分界线---------------------------------------------------------------")
    quantizer.set_hard()
    for epoch in range(25): # 第二组epochs优化离散的nbits
        x_dequant = quantizer(x)
        loss_quant = mse_loss(x, x_dequant)
        model_size, target_size = quantizer.model_size_loss()
        loss_size = mse_loss(model_size, target_size)
        loss = loss_size * k + loss_quant

        print(loss_size)
        print(loss_quant)
        print(quantizer.n_bits)
        optimizer.zero_grad()
        norm = loss_scaler(loss, optimizer, parameters=quantizer.n_bits)

    # 最后将所有的bit位round成整数
    # 其实不要这一步也可以，因为在第二组epochs时遵循的量化规律就是把n_bits round到
    # 整数了。
    quantizer.round_nbits()
    print(quantizer.n_bits)
    x_dequant = quantizer(x)

    


if __name__=="__main__":
    main()