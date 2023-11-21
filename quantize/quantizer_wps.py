"""
这是培松师兄给我的quantizer,我准备先好好理解它。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizationConfig:
    quan_type = None
    signed = None
    method = None
    bit_width = None
    scale = None
    max_val = None
    min_val = None
    def __init__(self, quan_type):
        assert(isinstance(quan_type, str))
        quan_type = quan_type.lower()
        self.quan_type = quan_type
        if quan_type.startswith('int'):                                                                                 # 均匀量化成有符号整数
            self.signed = True
            self.method = 'uniform'
            bit_width = quan_type[3:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)                                                                             # 比如说量化成3bit
            self.max_val = (1 << (self.bit_width - 1)) - 1                                                              # 那么max_val=3
            self.min_val = - self.max_val                                                                               # min_val=-3. 竟然不是-4.这样就损失了一个bin
        elif quan_type.startswith('uint'):                                                                              # 均匀量化成无符号整数
            self.signed = False
            self.method = 'uniform'
            bit_width = quan_type[4:]
            assert(bit_width.isdigit())                                                                 
            self.bit_width = int(bit_width)
            self.max_val = (1 << self.bit_width) - 1                                                                    # 对于3bit量化，量化到0，1，2，..., 7
            self.min_val = 0
        elif quan_type.startswith('log'):                                                                               # 对数量化，量化后成为有符号整数
            self.signed = True
            self.method = 'log'
            bit_width = quan_type[3:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << ((1 << (self.bit_width-1)) - 2)                                                         # 比如说3bit，量化后最大值是4，最小值是-4。那么这几个数是-4,-2,-1,0,1,2,4
            self.min_val = - self.max_val                                                                               # 如果是4bit量化，量化后最大值是2^6=64，最小值是-64. 对应的数是[-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64]
        elif quan_type.startswith('ulog'):                                                                              # 对数量化，量化后是无符号整数
            self.signed = False
            self.method = 'log'
            bit_width = quan_type[4:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << ((1 << self.bit_width) - 2)                                                             # 对3bit量化，量化后最大值是64，对应的数：[0,1,2,4,8,16,32,64]
            self.min_val = 0
        elif quan_type.startswith('oht'):                                                                               # 这是什么量化？
            self.signed = True  
            self.method = 'log'                                                                                         # 方法看起来还是对数量化
            bit_width = quan_type[3:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << (self.bit_width - 2)                                                                    # 如果是3bit，量化后最大值是2，最小值-2.
            self.min_val = - self.max_val
        elif quan_type.startswith('uoht'):
            self.signed = False
            self.method = 'log'
            bit_width = quan_type[4:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << (self.bit_width - 1)                                                                    # 如果是3bit，量化后最大值4，最小值0.
            self.min_val = 0
        elif quan_type == 'float32':
            pass
        else:
            assert(False)
        
    def activation_init_quantization(self, x, alpha=None):
        print("x.max", np.max(x))
        print("x.min", np.min(x))
        print("max_val", self.max_val)
        assert(np.min(x)>=0)

        circle_detection_queue = [0,]*5

        init_max_val = self.max_val
        if self.method == 'log':
            init_max_val = 8
        if alpha is None:
            alpha = np.max(np.fabs(x)) / init_max_val                                                                   # np.fabs(): 求绝对值
        alpha_old = alpha * 0   
        n_iter = 0
        circle_detection_queue[n_iter] = alpha
        while(np.sum(alpha!=alpha_old)):
            # print(alpha)
            q = x / alpha                                                                                               # 这里就是把x都scale到绝对值小于init_max_val了。
            if self.method == 'log':                                                                                    # 对于对数量化，
                q_0_idx = np.where(q<=0.5)
                q_thresh = np.log2(q/1.5) + 1
                q_thresh[q_thresh<0] = 0
                q = 2**(np.floor(q_thresh))
                # q = q * q_sign
                q[q_0_idx] = 0
                q = np.clip(q, self.min_val, self.max_val)
            elif self.method == 'uniform':                                                                              # 对于均匀量化，就是一个很简单的将x先scale到-min_val, max_val之间，然后
                q = np.clip(np.round(q), self.min_val, self.max_val)                                                    # 做一个round

            alpha_old = alpha;
            alpha = np.sum(x*q) / np.sum(q*q)                                                                           # 这里有一个alpha的更新步骤，这是一个收敛过程，找到最适合的scale

            if alpha in circle_detection_queue:
                break
            n_iter += 1
            circle_detection_queue[n_iter%5] = alpha
        return alpha


class WeightQuantizer:
    def __init__(self, model, quan_type, layer_type=(nn.Conv2d, nn.Linear), filterout_fn=None):
        self.quan_cfg = QuantizationConfig(quan_type)
        self.num_quan_layers = 0
        self.scales = []
        self.target_modules = []
        self.target_params= []
        self.saved_tensor_params = []
        index = -1
        for m_name, m in model.named_modules():
            if isinstance(m, layer_type):
                index += 1
                if filterout_fn is not None:
                    if filterout_fn(index, m_name, m):
                        continue
                self.target_modules.append(m)
                self.target_params.append(m.weight)
                self.saved_tensor_params.append(m.weight.data.clone())
                self.num_quan_layers += 1
                self.scales.append(None)

    def save_scales(self, f):
        torch.save(self.scales, f)

    def load_scales(self, f):
        self.scales = torch.load(f)

    def init_quantization(self):
        for index in range(self.num_quan_layers):
            s = self.target_params[index].data.size()
            w = self.target_params[index].data.view(s[0], -1)

            alpha = w.abs().max(dim=1)[0] / self.quan_cfg.max_val
            alpha_old = alpha * 1.1
            count = 0
            while((alpha-alpha_old).norm()>1e-9):
                q = self.quantize(w, alpha)
                alpha_old = alpha
                alpha = (w*q).sum(dim=1) / (q*q).sum(dim=1)
                count += 1
            self.scales[index] = alpha
            w.view(s)
            print(count)

    def quantize(self, w, alpha):
        q = w / alpha.unsqueeze(1)
        if self.quan_cfg.method == 'log':
            q_sign = q.sign()
            q_abs = q.abs()
            q_0_idx = q_abs<=0.5
            q_thresh = (q_abs/1.5).log2() + 1
            q_thresh[q_thresh<0] = 0
            q = 2**(q_thresh.floor())
            q = q * q_sign
            q[q_0_idx] = 0
            q.clamp_(self.quan_cfg.min_val, self.quan_cfg.max_val)
        elif self.quan_cfg.method == 'uniform':
            q.round_().clamp_(self.quan_cfg.min_val, self.quan_cfg.max_val)
        return q

    def quantization(self):
        #self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.quantizeConvParams()

    def clampConvParams(self):
        for index in range(self.num_quan_layers):
            s = self.target_params[index].data.size()
            w = self.target_params[index].data.view(s[0], -1)
            alpha = self.scales[index].unsqueeze(1)
            q = w / alpha
            q.clamp_(self.quan_cfg.min_val, self.quan_cfg.max_val)
            self.target_params[index].data.copy_((q*alpha).view(s))

    def save_params(self):
        for index in range(self.num_quan_layers):
            self.saved_tensor_params[index].copy_(self.target_params[index].data)

    def quantizeConvParams(self):
        for index in range(self.num_quan_layers):
            alpha = self.scales[index]
            s = self.target_params[index].data.size()
            w = self.target_params[index].data.view(s[0], -1)
            q = self.quantize(w, alpha)
            self.target_params[index].data.copy_((q*alpha.unsqueeze(1)).view(s))

    def restore(self):
        for index in range(self.num_quan_layers):
            self.target_params[index].data.copy_(self.saved_tensor_params[index])

class QuantizeF(torch.autograd.Function):

    @staticmethod
    def forward(self, input2, quan_cfg):
        self.save_for_backward(input2)
        self.quan_cfg = quan_cfg

        if quan_cfg.scale is None: #without quantization
            output = input2
        else: #with quantization
            if quan_cfg.method == 'uniform':
                output = (input2 / quan_cfg.scale).round().clamp(quan_cfg.min_val, quan_cfg.max_val) * quan_cfg.scale   # fake quant，因为最后又把scale乘上了。
            elif quan_cfg.method == 'log':
                q = input2 / quan_cfg.scale                                                                             # 先除以scale，使q在0到64之间。

                # output = q.clone()
                # output[q<=0.5] = 0.0
                # output[(q>0.5)&(q<=1.5)] = 1.0
                # output[(q>1.5)&(q<=3)] = 2.0
                # output[(q>3)&(q<=6)] = 4.0
                # output[(q>6)&(q<=12)] = 8.0
                # output[(q>12)&(q<=24)] = 16.0
                # output[(q>24)&(q<=48)] = 32.0
                # output[(q>48)] = 64.0
                # outputa = output.clone()
                # #print(outputa)
                # output = output * quan_cfg.scale

                q_sign = q.sign()
                q_abs = q.abs()
                # q_0_idx = torch.where(q_abs<=0.5)
                q_0_idx = q_abs<=0.5                                                                                    # <=0.5的值的索引
                q_thresh = (q_abs/1.5).log2() + 1                                                                       # 对于0.5到64之间的数，这样的操作能保证正确的量化吗？
                q_thresh[q_thresh<0] = 0
                output = 2**(q_thresh.floor())
                output = output * q_sign
                output[q_0_idx] = 0
                output.clamp_(quan_cfg.min_val, quan_cfg.max_val).mul_(quan_cfg.scale)
            else:
                assert(False)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        quan_cfg = self.quan_cfg
        grad_input = grad_output.clone()
        if quan_cfg.scale is not None: #without quantization
            grad_input[input.ge(quan_cfg.max_val * quan_cfg.scale)] = 0                 # 看起来像是比max_val * scale大的部分的梯度直接为0
            grad_input[input.le(quan_cfg.min_val * quan_cfg.scale)] = 0                 # 比 min_val * scale小的部分也是梯度为0，这里实际上是应对clamp操作
        return grad_input, None                                                         # 其余部分的梯度执行ste


class Quantization(nn.Module):
    def __init__(self, quan_type):
        super(Quantization, self).__init__()
        self.quan_cfg = QuantizationConfig(quan_type)
        self.quan_fn = QuantizeF.apply #(self.quan_cfg)

    def forward(self, input):
        return self.quan_fn(input, self.quan_cfg)
        # return QuantizeF.apply(input, self.quan_cfg)

    def set_scale(self, scale):
        self.quan_cfg.scale = scale
        
    def init_quantization(self, X, alpha=None):
        alpha = self.quan_cfg.activation_init_quantization(X, alpha)
        return alpha

def main():
    return 

if __name__ == "__main__":
    main()