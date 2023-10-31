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



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
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
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
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

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
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
        x_int = x_int.clamp(self.qmin, self.qmax)
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

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)
        
        if self.dynamic:
            if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":                              # 权重到这里就量化完了
                self.per_token_dynamic_calibration(x)
        elif self.mode == "calibration":                                                                                # 这是针对激活值的，它可能是3D（1,2048,768)或者2D(2048,768)
            self.calibration(x)
            # return x                                                                                                  # 这里不像rptq，即便是在calibration的时候也要对激活值做fake_quant再返回
        else:                                                                                                           # 主要是对activation，针对eval模式，直接fake_quant
            pass

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)

        return x_dequant
    """
    weight量化永远是dynamic(假的), per_token.
    而activation量化只要不是dynamic, 就有两种可能, per_tensor和per_cluster. 我们就在per_cluster_minmax_calibration()
    函数里面做文章, 加入rptq的那些按类量化的机制
    """
    """
    A example:
    对opt-125m模型, k_proj这个权重矩阵, 在一个epoch里面执行临时性的量化时,
    X形状为[768,768], 它的每个output_channel得到一个最大值与最小值, 从而得到scale, 形状为[768,1]
    A example:
    opt-125m模型, q_proj的输入为[1,2048,768], self.dynamic_method="per_token", xmin, xmax, scale形状都是[1, 2048, 1], 表示每个token一个scale, 
    这是比较硬件友好的。
    """

    def calibration(self, x: torch.Tensor):
        if self.dynamic_method == "per_tensor":
            scale, zero_point = self.per_tensor_minmax_calibration(x)
        elif self.dynamic_method == "per_cluster":
            scale, zero_point = self.per_cluster_minmax_calibration(x)
        del self.scale
        self.register_buffer("scale", scale)

        zero_point.clamp_(min=-1e4, max=1e4)
        del self.zero_point, self.round_zero_point
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("round_zero_point", zero_point.round())

    def per_cluster_minmax_calibration(self, x:torch.Tensor):                                                           
        """
        只针对激活值做量化, 激活值可能是2D或者3D
        此时self.cluster_dim和self.cluster_counts都是有东西的
        """
        reduce_axes = [
            _
            for _ in range(x.ndim)
            if _ not in self.per_channel_axes and _ != self.cluster_dim
        ]
        """
        比如x是个3维张量, 那么x.ndim=3,
        经过reorder后的self.cluster_dim=2,
        self.per_channel_axes=[],
        那么reduce_axes就是[0,1], 表示这两个维度被reduce掉,
        从而获得相应的scale, zero_point
        """
        if self.symmetric:
            pass                    # 激活值量化不宜使用对称
        else:
            xmin = x.amin(reduce_axes, keepdim=True).clone()
            xmax = x.amax(reduce_axes, keepdim=True).clone()
            if self.cached_xmax is not None:
                if self.metric == "minmax":
                    xmax = torch.max(self.cached_xmax, xmax)
                    xmin = torch.min(self.cached_xmin, xmin)
                if self.metric == "ema_minmax":
                    xmax = self.cached_xmax * 0.99 + xmax * 0.01
                    xmin = self.cached_xmin * 0.99 + xmin * 0.01
            self.cached_xmax = xmax
            self.cached_xmin = xmin
            if self.cluster_dim is not None:        # 显然成立
                st = 0
                for count in self.cluster_counts:
                    part_xmin = torch.narrow(xmin, self.cluster_dim, st, count).clone()
                    part_xmax = torch.narrow(xmax, self.cluster_dim, st, count).clone()
                    cluster_xmin = part_xmin.amin(self.cluster_dim, keepdim=True)
                    cluster_xmax = part_xmax.amax(self.cluster_dim, keepdim=True)
                    xmin.narrow(self.cluster_dim, st, count).copy_(cluster_xmin)
                    xmax.narrow(self.cluster_dim, st, count).copy_(cluster_xmax)
                    st += count
            scale = (xmax - xmin) / (2 ** self.n_bits - 1)
            scale.clamp_(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (scale)
            return scale, zero_point


    
    def per_tensor_minmax_calibration(self, x:torch.Tensor):
        if self.symmetric:
            pass                    # 激活值量化不宜使用对称
        else:
            xmin = torch.min(x)
            xmax = torch.max(x)
            if self.cached_xmax is not None:
                if self.metric == "minmax":
                    xmax = torch.max(self.cached_xmax, xmax)
                    xmin = torch.min(self.cached_xmin, xmin)
                if self.metric == "ema_minmax":
                    xmax = self.cached_xmax * 0.99 + xmax * 0.01
                    xmin = self.cached_xmin * 0.99 + xmin * 0.01
            self.cached_xmax = xmax
            self.cached_xmin = xmin

            scale = (xmax - xmin) / (2 ** self.n_bits - 1)
            scale.clamp_(min=CLIPMIN, max=1e4)
            # zero_point = (xmax + xmin) * (-0.5 / scale)
            zero_point = -(xmin) / (scale)
            return scale, zero_point

            
    def set_calibration_mode(self):
        self.mode = "calibration"

    def set_eval_mode(self):
        self.mode = "eval"


    def per_token_dynamic_calibration(self, x):
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
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            self.scale = scale
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)                                                                      # 注册到buffer里面，之后移动device的时候就很方便了
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def free(self):
        self.cached_xmax = None
        self.cached_xmin = None