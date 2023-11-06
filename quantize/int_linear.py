import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
from quantize.quantizer_exp import ExpAffineQuantizer
from quantize.quantizer_mix import MixAffineQuantizer
import numpy as np





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        weight_exp_quant=False,
        weight_mix_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        if weight_exp_quant:
            self.weight_quantizer = ExpAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        elif weight_mix_quant:
            self.weight_quantizer = MixAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        else:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        
        """
        A example: 
        opt-125m, layer 0, k_proj.
        It has bias.
        self.in_features = self.out_features = 768
        disable_input_quant = false. It means k_proj also need to quantize input!
        """
    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out
    """
    A example:
    q_proj的输入: [1,2048,768]
    """

    def set_ic_cluster_counts(self, counts, w_dim=1, a_dim=2):                                                          # ltl添加函数
        self.weight_quantizer.cluster_dim = w_dim                                                                       # 这两行其实没用
        self.weight_quantizer.cluster_counts = counts                                                                   # 因为weight直接采用per_output_channel的量化了
        if a_dim is not None:
            self.act_quantizer.cluster_dim = a_dim
            self.act_quantizer.cluster_counts = counts


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
