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
from quantize.reorder_utils import (
    R1_reorder,
    R2_reorder,
    R3_reorder,
    R4_reorder,
    R5_reorder,
    layer_i0max_hook,
    layer_omax_hook,
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
)


def Reorder(qlayer, config, enable_R1, enable_R2, enable_R3, enable_R4, enable_R5, n_clusters, dev):
    indices = {}
    if not enable_R1:
        counts = np.ones(int(n_clusters["R1"]), dtype=int) * (config.hidden_size / n_clusters["R1"])
        counts = counts.astype(int)
        R1_index = torch.arange(config.hidden_size).to(dev)
        R1_reorder(
            qlayer.self_attn_layer_norm, 
            qlayer.self_attn.q_proj,
            qlayer.self_attn.k_proj,
            qlayer.self_attn.v_proj,
            R1_index,
            counts,
        )
    else:
        feature_max, feature_min = oc_maxmin_dict[f"self_attn_layer_norm"]
        R1_index, counts = tensor_calc_reorder_index(
            feature_max, feature_min, n_clusters["R1"]
        )
        R1_reorder(
            qlayer.self_attn_layer_norm,
            qlayer.self_attn.q_proj,
            qlayer.self_attn.k_proj,
            qlayer.self_attn.v_proj,
            R1_index,
            counts,
        )

    if not enable_R2:
        num_head = config.num_attention_heads
        clusters = n_clusters["R2"] * num_head
        counts = np.ones(int(clusters), dtype=int) * config.hidden_size / clusters
        counts = counts.astype(int)
        R2_index = torch.arange(config.hidden_size).to(dev)
        R2_reorder(
            qlayer.self_attn.q_proj,
            qlayer.self_attn.k_proj,
            qlayer.self_attn.qkt_matmul,
            R2_index,
            counts,
        )
    else:
        qmax, qmin = oc_maxmin_dict[f"self_attn.q_proj"]
        kmax, kmin = oc_maxmin_dict[f"self_attn.k_proj"]
        R2_index, counts = tensor_calc_reorder_index(
            [qmax, kmax], [qmin, kmin], n_clusters["R2"], qlayer.self_attn.num_heads
        )
        # print("R2 index counts", counts)
        R2_reorder(
            qlayer.self_attn.q_proj,
            qlayer.self_attn.k_proj,
            qlayer.self_attn.qkt_matmul,
            R2_index,
            counts,
        )

    if not enable_R3:
        num_head = config.num_attention_heads
        clusters = n_clusters["R3"] * num_head
        counts = np.ones(int(clusters), dtype=int) * config.hidden_size / clusters
        counts = counts.astype(int)
        R3_index = torch.arange(config.hidden_size).to(dev)
        R3_reorder(
            qlayer.self_attn.v_proj,
            qlayer.self_attn.pv_matmul,
            qlayer.self_attn.out_proj,
            R3_index,
            counts,
        )
    else:
        feature_max, feature_min = ic_maxmin_dict[f"self_attn.out_proj"]
        R3_index, counts = tensor_calc_reorder_index(
            feature_max, feature_min, n_clusters["R3"], qlayer.self_attn.num_heads
        )
        # print("R3 index counts", counts)
        R3_reorder(
            qlayer.self_attn.v_proj,
            qlayer.self_attn.pv_matmul,
            qlayer.self_attn.out_proj,
            R3_index,
            counts,
        )

    if not enable_R4:
        counts = np.ones(int(n_clusters["R4"]), dtype=int) * (config.hidden_size / n_clusters["R4"])
        counts = counts.astype(int)
        R4_index = torch.arange(config.hidden_size).to(dev)              
        R4_reorder(
        qlayer.final_layer_norm,
        qlayer.fc1,
        R4_index,
        counts,
    )
    else:
        feature_max, feature_min = oc_maxmin_dict[f"final_layer_norm"]

        R4_index, counts = tensor_calc_reorder_index(
            feature_max, feature_min, n_clusters["R4"]
        )
        # print("R4 index counts", counts)
        R4_reorder(
            qlayer.final_layer_norm,
            qlayer.fc1,
            R4_index,
            counts,
        )

    if not enable_R5:
        counts = np.ones(int(n_clusters["R5"]), dtype=int) * (config.ffn_dim) / n_clusters["R5"]
        counts = counts.astype(int)
        R5_index = torch.arange(config.ffn_dim).to(dev)
        R5_reorder(
        qlayer.fc1,
        qlayer.fc2,
        R5_index,
        counts,
    )
    else:
        feature_max, feature_min = ic_maxmin_dict[f"fc2"]
        R5_index, counts = tensor_calc_reorder_index(
            feature_max, feature_min, n_clusters["R5"]
        )
        # print("R5 index counts", counts)
        R5_reorder(
            qlayer.fc1,
            qlayer.fc2,
            R5_index,
            counts,
        )
    indices["R1"] = R1_index
    indices["R2"] = R2_index
    indices["R3"] = R3_index
    indices["R4"] = R4_index
    indices["R5"] = R5_index
    return indices

def register_hooks(layer, enable_R1, enable_R2, enable_R3, enable_R4, enable_R5):
    handlers = []
    for name, module in layer.named_modules():
        if(
            enable_R1
            and isinstance(module, nn.LayerNorm)
            and "attn_layer_norm" in name
        ):
            module.name = name
            handler = module.register_forward_hook(layer_omax_hook)
            handlers.append(handler)
        if (
            enable_R2
            and isinstance(module, nn.Linear)
            and ("q_proj" in name or "k_proj" in name)
        ):
            module.name = name
            handler = module.register_forward_hook(layer_omax_hook)
            handlers.append(handler)
        if (
            enable_R3 
            and isinstance(module, nn.Linear) 
            and "out_proj" in name
        ):
            module.name = name
            handler = module.register_forward_hook(layer_i0max_hook)
            handlers.append(handler)
        if (
            enable_R4
            and isinstance(module, nn.LayerNorm)
            and "final_layer_norm" in name
        ):
            module.name = name
            handler = module.register_forward_hook(layer_omax_hook)
            handlers.append(handler)
        if (
            enable_R5 
            and isinstance(module, nn.Linear) 
            and "fc2" in name
        ):
            module.name = name
            handler = module.register_forward_hook(layer_i0max_hook)
            handlers.append(handler)
    return handlers


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # 1. 得到layers[0]的输入
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache                                              
    model.config.use_cache = False                                                                          # 校验的时候都不use cache, 包括推理的时候也不。因为这些时候都是一次输入2048个token，而不是一次输入一个token再输出一个token。
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
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
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    
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
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input             # None
    out_temp = None
    
    attention_mask = cache["attention_mask"]                                                                            # [1, 1, 20148, 2048]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float() # [1, 1, 2048, 2048]
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    reorder = args.reorder
    enable_clusters = True if "0" in reorder else False
    enable_R1 = True if "1" in reorder else False
    enable_R2 = True if "2" in reorder else False
    enable_R3 = True if "3" in reorder else False
    enable_R4 = True if "4" in reorder else False
    enable_R5 = True if "5" in reorder else False
    n_clusters = {
        "R1": args.R1_clusters,
        "R2": args.R2_clusters,
        "R3": args.R3_clusters,
        "R4": args.R4_clusters,
        "R5": args.R5_clusters,
        }

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
    # 2. 开始对每个layer做量化
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)                                                                                         # 这时候还是全精度模型



        # 2.1 reorder部分
        if enable_clusters:                                                                                             
            # 2.1.1 注册前向钩子
            handlers = register_hooks(layer, enable_R1, enable_R2, enable_R3, enable_R4, enable_R5)
                
            # 2.1.2 前向过程。注意这里的输入是quant_inps而不是fp_inps，因为我就是按照RPTQ上面来的
            for j in range(args.nsamples):
                out_temp = layer(
                    quant_inps[j].unsqueeze(0).to(dev).to(torch.float16), attention_mask=attention_mask.to(dev)
                )[0]
            for handler in handlers:
                handler.remove()


            # 2.1.3 进行重排序
            indices = Reorder(qlayer, model.config, enable_R1, enable_R2, enable_R3, enable_R4, enable_R5, n_clusters, dev)








        # 2.2 OmniQuant部分
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]      # 这一个全精度block的输出，很好，是reorder之后的模型，这样可以使得搜索过程专注于omni 
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # if is_llama and args.abits == 16:
        #     use_shift = False                   # deactivate channel-wise shifting for llama weight-
        # use_shift = True if args.abits < 16 else False   # only activate per-channel shifting when weight-activation quantization
        
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            if enable_clusters:                                                                         # 对act和shift也要进行reorder
                                if key == "q_proj":
                                    act = torch.index_select(act, 0, indices["R1"])
                                if key == "out_proj":
                                    act = torch.index_select(act, 0, indices["R3"])
                                if key == "fc1":
                                    act = torch.index_select(act, 0, indices["R4"])
                            weight = module.weight.max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                                if enable_clusters:                                                                     # 对act和shift也要进行reorder
                                    if key == "q_proj":
                                        shift = torch.index_select(shift, 0, indices["R1"])
                                    if key == "out_proj":
                                        shift = torch.index_select(shift, 0, indices["R3"])
                                    if key == "fc1":
                                        shift = torch.index_select(shift, 0, indices["R4"])
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
        """
        到这里得到了一个block中所有线性层的shift和scale, 形状都是[768]. 另外还有qkt_smooth_scale.
        """
        
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":qlayer.let_parameters(use_shift),"lr":args.let_lr}, {"params":qlayer.lwc_parameters(),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            """
            其中, let_parameters一共有7个数组, 大小都是[768]. 第一个是qkt的scale,初始化为全1。
            第2,3个是q_proj的输入scale, shift.
            第4, 5个是out_proj的输入scale, shift.
            第6,7个是fc1的输入scale, shift.
            lwc_paramters一共有12个数组, 10个大小为[768,1], 另外两个为[3072,1]. 它们来自所有被建立了quantizer的权重矩阵, 
            有自己的upbound_factor和down_bound_factor. 一共有6个权重矩阵。
            """

            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary()                                                                                                               
                        """     
                        到这里是smooth & shift了两个layer_norm及后面的4个线性层, v_proj及后面的out_proj, q_proj和k_proj, 以及把所有的权重矩阵都量化了
                        """
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0] # 该block量化后的输出
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)                              # 全精度模型的输出和量化后模型的输出作差求loss函数。
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.omni_parameters(use_shift))                    # 看样子这里是更新参数
                    norm_list.append(norm.data)                                                                         # 在不断地更新优化器的参数
                    
                    """
                    需要清空所有a_quantizer的cached_max, cached_min
                    """
                    for name, m in qlayer.named_modules():
                        if isinstance(m, (QuantLinear)) and not m.disable_input_quant:
                            m.act_quantizer.free()
                        if isinstance(m, QuantMatMul):
                            m.x1_quantizer.free()
                            m.x2_quantizer.free()

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            qlayer.clear_temp_variable()
            del optimizer

        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()                                                                               # 这里就是把qlayer所有的权重矩阵/layernorm层参数用优化好了的那些scale, shift参数来重新计算一遍
        if args.epochs>0:                                                                                               # 其中, 权重矩阵都要做per_output_channel的量化
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0] # 得到这一层量化后的输出
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = qlayer.omni_state_dict()
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))                                          
            # 保存了omni的参数，方便下次调用。比如说这次训练了20个epoch，下次要训练40个epochs的时候就只用
            # 加载这次的参数，然后再训练20个epochs
        else:
            qlayer.half()
            for j in range(args.nsamples):
                quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            qlayer.register_scales_and_zeros()
            layers[i] = qlayer.to("cpu")
        # 然后就要设置所有的a_quantizer的mode=eval
        for name, m in qlayer.named_modules():
            if isinstance(m, (QuantLinear)) and not m.disable_input_quant:
                m.act_quantizer.set_eval_mode()
            if isinstance(m, QuantMatMul):
                m.x1_quantizer.set_eval_mode()
                m.x2_quantizer.set_eval_mode()


        
        if args.real_quant:                                                                                             # 我目前没做什么real_quant
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.float().cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                levels = name.split('.')
                if len(levels) > 1:
                    mod_ = qlayer
                    for l_idx in range(len(levels)-1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], q_linear)
                else:
                    setattr(qlayer, name, q_linear)        
                del module        
        del layer
        # del fp_inps
        # fp_inps= copy.deepcopy(quant_inps)                                                                              # 会怎样，事实证明结果更差！
        """
        fp_inps就一直是浮点数模型的输入输出, quant_inps也一直是量化后模型的输入输出, 有权重量化和激活值量化
        """
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

