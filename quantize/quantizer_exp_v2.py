"""
之前我写的指数量化代码，没有量化为0的那一项。现在我考虑加上0.
参考的是quantizer_wps.py。
对于3bit量化，先将量化后的值表示为0,1,2,4,8,16,32,64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5


class FakeQuant(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, scale, nbit):
        # x: [768,768]
        # scale: [768,1]
        # nbit: 暂时还是个标量
        ctx.qmin = - 1 << ((1 << nbit-1) - 2)                                                               # 对3bit量化，就是-4, -2, -1, 0, 1, 2, 4, 一共七个数
        ctx.qmax = 1 << ((1 << nbit-1) - 2)  
        
        # 1. 除以scale
        q = x / scale

        # 2. 保留符号，并取绝对值
        sign = q.sign()
        q_abs = q.abs()
        q_0_idx = q_abs < 0.5
        q_abs = q_abs.clamp(1e-4, 1e4)                                                                                  # weight不可能有1e4那么大，但是可能小于1e-4.不过我暂时不处理这方面的梯度

        # 3. 求对数了
        q_thresh = torch.log2(q_abs / 1.5) + 0.5
        q_thresh_idx = q_thresh < -0.5
        q_thresh[q_thresh_idx] = 0

        # 4. floor操作，相当于-0.5然后再round,以及我把-0.5挪到前面一步里面了
        q_floor = q_thresh.round()

        # 5. 指数操作
        q_exp = 2 ** q_floor

        # 最重要的一步：让q_0_idx部分的值为0
        q_exp[q_0_idx] = 0

        # clamp到qmin和qmax之间
        q_clamp = q_exp.clamp(ctx.qmin, ctx.qmax)

        # 乘以符号
        q_sign = q_clamp * sign             # 对于3bit量化，到这里就只看到了-4,-2,-1,0,1,2,4, 算是成功

        # 5. 最后再乘以scale
        x_dequant = q_sign * scale

        ctx.save_for_backward(x, scale, sign, q_abs, q_exp, q_sign, q_0_idx, q_thresh_idx)
        return x_dequant
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        # print("成功的计算梯度")           # 确实打印出来了！
        input, scale, sign, q_abs, q_exp, q_sign, q_0_idx, q_thresh_idx = ctx.saved_tensors
        # 对x_dequant的导数
        grad_output = grad_output.expand(input.shape)                                    # 扩展一下，否则算不了


        # q_sign的导数和scale的第一部分导数
        grad_q_sign = grad_output * scale
        grad_scale_1 = grad_output * q_sign

        # q_clamp的导数
        grad_q_clamp = grad_q_sign
        grad_q_clamp[sign == -1] = grad_q_clamp[sign == -1] * -1

        # q_exp的导数
        grad_q_clamp[q_exp.ge(ctx.qmax+0.0001)] = 0
        grad_q_clamp[q_exp.le(ctx.qmin+0.0001)] = 0
        grad_q_clamp[q_0_idx] = 0 
        grad_q_exp = grad_q_clamp

        # q_floor的导数
        grad_q_floor = math.log(2) * q_exp * grad_q_exp

        # q_thresh的导数
        grad_q_thresh = grad_q_floor
        grad_q_thresh[q_thresh_idx] = 0

        # q_abs的导数
        grad_q_abs = grad_q_thresh / math.log(2) / q_abs

        # q的导数
        grad_q = grad_q_abs * sign

        # input的导数
        grad_input = grad_q / scale

        # scale的导数
        grad_scale = grad_scale_1 + grad_q * (-input / scale ** 2)

        return grad_input, grad_scale, None


class ExpAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        target_bit=0,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.n_bits = n_bits
        if symmetric:
            self.qmin = - 1 << ((1 << self.n_bits-1) - 2)                                                               # 对3bit量化，就是-4, -2, -1, 0, 1, 2, 4, 一共七个数
            self.qmax = 1 << ((1 << self.n_bits-1) - 2)                                                                 # 对4bit量化, 就是0,1,2,4,8,16,32,64,带上+-号，有15个数
        else:
            print("指数量化不支持非对称！")

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

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 1 << ((1 << self.n_bits-1) - 2) 

    def forward(self, x: torch.Tensor):
        self.per_channel_dynamic_calibration(x)
        x_dequant = FakeQuant.apply(x, self.scale, self.n_bits)

        if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
            breakpoint()

        if torch.isinf(x_dequant).any() or torch.isnan(x_dequant).any():
            breakpoint()

        return x_dequant

    def per_channel_dynamic_calibration(self, x):
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

        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (self.qmax - self.qmin)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.zero_point = None

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)                                                                      # 注册到buffer里面，之后移动device的时候就很方便了
        self.register_buffer('zeros', self.zero_point)
        del self.scale
        del self.round_zero_point


def main():
    torch.manual_seed(10)
    X = torch.randn(1000, 1000)
    quantizer = ExpAffineQuantizer(n_bits=4, symmetric=True)
    x_dequant = quantizer(X)
    loss = torch.mean(torch.abs(X - x_dequant))

    print(X[0,0:20])
    print(x_dequant[0,0:20])
    print(loss)

if __name__ == "__main__":
    main()