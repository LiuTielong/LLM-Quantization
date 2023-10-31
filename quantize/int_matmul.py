import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant
    
        """ 
        A example: 
        opt-125m, w3a16, layer 0, qkt_quant.
        x1_quant_params=x2_quant_params, 非对称, per_channel_axes=[], dynamic_method="per_token"
        """


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out
    
    def set_ic_cluster_counts(                                                                                          # ltl添加函数
            self, counts, x1_dim=2, x2_dim=2, cluster_x1=True, cluster_x2=True
    ):
        if cluster_x1:
            self.x1_quantizer.cluster_dim = x1_dim
            self.x1_quantizer.cluster_counts = counts
        if cluster_x2:
            self.x2_quantizer.cluster_dim = x2_dim
            self.x2_quantizer.cluster_counts = counts
