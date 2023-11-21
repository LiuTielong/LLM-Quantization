"""
本文件用来获得所有block的所有output channel的最大eigenvalue，然后作为一个文件存起来。
每个block都有几千个值，一整个模型应该有几万个数。

"""
import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
import numpy as np
from math import inf
import csv
import pdb
torch.manual_seed(10)

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    # return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])                                                       # zip：将多个可迭代对象（例如列表、元组等）的对应元素打包成一个元组的序列
    if isinstance(xs, list):
        return [torch.sum(x * y, dim=-1, keepdim=True) for (x, y) in zip(xs, ys)]
    else:
        return torch.sum(xs * ys, dim=-1, keepdim=True) 

def de_variable(v):
    '''
    normalize the vector and detach it from variable
    '''

    s = group_product(v, v)
    if isinstance(v, list):
        s = [si**0.5 for si in s]                                                                                                      # 这个s就是二范数
        v = [vi / si for (vi, si) in zip(v, s)]
    else:
        s = s**0.5
        v = v / s
    return v

def get_grads(
    lm, 
    args,
    dataloader,
):
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache                                              
    model.config.use_cache = False                                                                          # 校验的时候都不use cache, 包括推理的时候也不。因为这些时候都是一次输入2048个token，而不是一次输入一个token再输出一个token。
    is_llama = False
    if "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"

    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
    traincast = torch.cuda.amp.autocast

    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    quant_inps = inps                                                                                                   # [128, 2048, 768]
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps2 = copy.deepcopy(inps)

    attention_mask = cache["attention_mask"]                                                                            # [1, 1, 20148, 2048]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float() # [1, 1, 2048, 2048]
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    
    args.weight_quant_params = {
        "n_bits":16,
        "per_channel_axes": [0],
        # "symmetric": args.symmetric if not args.weight_exp_quant else True,                                                 # 看来对权重用点对称量化问题不大，对激活值坚决不能用
        "symmetric": args.w_symmetric,
        "dynamic_method": args.w_dynamic_method,                                                                        # 对权重采用per-channel的量化吗
        "group_size": args.group_size,
        "lwc":args.lwc,                                                                                                 # 权重相比于激活值，有这么个参数
        "dynamic": True,                                                                                                # 权重的dynamic是true，只是为了对它进行per_channel的量化，不是真的动态
        "metric": "minmax",
        "target_bit":args.wbits,
    }
    csv_file = open("results/hawq_values.csv", 'w', newline='')
    writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['block', 'iters', 'max_eigenvalue'])



    for i in range(len(layers)):
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)  
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        qlayer.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for j in range(args.nsamples):
                    fp_inps2[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0] 

        qlayer.train()

        with torch.no_grad():
            qlayer.float()
        qlayer.set_quant_state(weight_quant=False, act_quant=True)
        
        percentage = 1.0                                                                    # 多少百分比的calib data拿去算梯度，我们希望是100%。
        acc_mean_grads = [torch.zeros(qlayer.fc1.weight.shape[0]).to(dev), torch.zeros(qlayer.fc2.weight.shape[0]).to(dev), 
                        torch.zeros(qlayer.self_attn.q_proj.weight.shape[0]).to(dev),torch.zeros(qlayer.self_attn.k_proj.weight.shape[0]).to(dev),
                        torch.zeros(qlayer.self_attn.v_proj.weight.shape[0]).to(dev), torch.zeros(qlayer.self_attn.out_proj.weight.shape[0]).to(dev)]
        for j in range(args.nsamples//args.batch_size):
            with traincast():
                index = j * args.batch_size
                if index > args.nsamples * percentage:
                    break
                quant_out = qlayer(fp_inps[index:index+args.batch_size,], 
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids
                                )[0]
                fp_inps_batch = fp_inps2[index:index+args.batch_size,]
                loss = torch.nn.functional.mse_loss(fp_inps_batch, quant_out) * 1e1
                loss.backward()
                grads = [qlayer.fc1.weight.grad, qlayer.fc2.weight.grad, qlayer.self_attn.q_proj.weight.grad,
                            qlayer.self_attn.k_proj.weight.grad, qlayer.self_attn.v_proj.weight.grad, qlayer.self_attn.out_proj.weight.grad]
                for grad, acc_mean_grad in zip(grads, acc_mean_grads):
                    acc_mean_grad += torch.mean(torch.abs(grad), dim=1)
                qlayer.zero_grad()
            """
            下面这一段free是要命的重要。否则出现经典问题:
            RuntimeError: Trying to backward through the graph a second time (or directly 
            access saved tensors after they have already been freed). Saved intermediate values 
            of the graph are freed when you call .backward() or autograd.grad(). Specify
            retain_graph=True if you need to backward through the
            graph a second time or if you need to access saved tensors after calling backward.
            """
            for name, m in qlayer.named_modules():
                if isinstance(m, (QuantLinear)) and not m.disable_input_quant:
                    m.act_quantizer.free()
                if isinstance(m, QuantMatMul):
                    m.x1_quantizer.free()
                    m.x2_quantizer.free()

        # 设置10%的channels高bit，10%低bit。看起来就是v_proj和o_proj的bit位比较高一些。
        acc_mean_grads_all = torch.cat(acc_mean_grads, dim=0)               # 长度为6912的向量
        des_mean_grads_all, indices = torch.sort(acc_mean_grads_all, descending=True)
        n_bits = torch.zeros_like(acc_mean_grads_all)
        len_all = acc_mean_grads_all.shape[0]
        n_bits[indices[0:int(len_all * 0.1)]] = 4
        n_bits[indices[int(len_all * 0.1): int(len_all * 0.9)]] = 3
        n_bits[indices[int(len_all * 0.9) :]] = 2
        torch.save(n_bits, f"results/layer{i}_nbits.pt")

        # qlayer.eval()
        # qlayer.set_quant_state(weight_quant=False, act_quant=False)
        # with torch.no_grad():
        #         with torch.cuda.amp.autocast():
        #             for j in range(args.nsamples):
        #                 quant_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0] # 得到这一层量化后的输出
        fp_inps = fp_inps2

        del layer
        del qlayer


        


        

