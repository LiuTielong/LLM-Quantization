import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

"""
指数量化, 为权重在极低bit量化时而生, 只支持2,3,4 bit权重量化。
权重符合一个类正态分布，所以我只支持对称的指数量化。
"""

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
        lwc=False
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        # assert symmetric == True, "指数量化只支持对称的"
        assert 2 <= n_bits <= 4, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1) - 1)        # 因为有一个bit位要用来存符号
        self.qmax = 0
        # 对于3bit量化，q的取值就是0，-1，-2，-3

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
        self.qmin = -(2 ** (n_bits - 1) - 1)        # 因为有一个bit位要用来存符号
        self.qmax = 0


    def fake_quant(self, x, scale, zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        # 假设是3bit量化，那么除去符号位还有2bit，可以表示0,-1,-2,-3
        # 那么x量化后的取值就可能是 +- 2^0, 2^-1, 2^-2, 2^-3
        # dequant之后没办法将x量化到0点？先不管！最小的数就是2^-3了。
        if zero_point is not None:
            x = x - zero_point                                                                              # 1. 将x作平移，变成对称分布
        sign = torch.sign(x)                                                                            # 2. 获得符号，得到x的绝对值
        x_abs = torch.abs(x / scale)                                                                    # 3. 缩放，将所有的数映射到[0,1]之间
        x_abs = x_abs.clamp(CLIPMIN, 1e4)                                                               # 4. 保护为0的值
        x_log = torch.log2(x_abs)                                                                       # 5. 比如x_log=-3.1，表示这个数和2^-3接近
        x_int = round_ste(x_log)                                                                        # 6. 得到-3
        x_int = x_int.clamp(self.qmin, self.qmax)                                                       # 7. 裁剪一下
        
        x_dequant = x_int
        x_dequant = torch.pow(2.0, x_dequant)                                                             # 8. 又变到2^-3
        x_dequant = x_dequant * sign * scale
        if zero_point is not None:
            x_dequant = x_dequant + zero_point

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        self.per_channel_dynamic_calibration(x)
        x_dequant = self.fake_quant(x, self.scale, self.zero_point)

        if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
            breakpoint()

        if torch.isinf(x_dequant).any() or torch.isnan(x_dequant).any():
            breakpoint()

        # loss = torch.mean(torch.abs(x - x_dequant))
        # print(loss)

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
            
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max                                     # 这样，基本上就把所有的数映射到[0,1]之间了
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            self.zero_point = None
        else:
            scale = (xmax - xmin) / 2
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            self.zero_point = (xmax + xmin) / 2 
        

        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)                                                                      # 注册到buffer里面，之后移动device的时候就很方便了
        self.register_buffer('zeros', self.zero_point)
        del self.scale
        del self.round_zero_point


def main():
    torch.manual_seed(10)
    X = torch.randn(1000, 1000)
    quantizer = ExpAffineQuantizer(n_bits=4, symmetric=False)
    x_dequant = quantizer(X)
    loss = torch.mean(torch.abs(X - x_dequant))

    print(X[0,0:20])
    print(x_dequant[0,0:20])
    print(loss)

if __name__ == "__main__":
    main()