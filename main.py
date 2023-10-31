import os
import sys
import random
import numpy as np
from models.opt import OPTClass
from models.llama import LLAMAClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

import pdb


# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5"

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "llama-2-7b",
    "llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
]


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args.eval_ppl:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:     # ltl
        for dataset in ["wikitext2", "ptb", "c4"]:
            cache_testloader = f'{args.calib_cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                    cache_dir=args.model_cache_dir
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", default="../LLMs", help="where to load the original model")
    parser.add_argument("--calib_cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    # parser.add_argument("--net", type=str, default="llama-2-13b", choices=net_choices)
    # parser.add_argument("--output_dir", default="./log/llama-2-7b-w3a16", type=str, help="direction of logging file")
    parser.add_argument("--net", type=str, default="opt-125m", choices=net_choices)
    parser.add_argument("--output_dir", default="./log/opt-125m-w3a16", type=str, help="direction of logging file")
    parser.add_argument("--epochs", type=int, default=0)                                                                # ltl
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true",)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", default=True, action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=3)                                                                     # ltl
    parser.add_argument("--abits", type=int, default=8)                                                                    # ltl
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--let", action="store_true",
                        default=True,
                         help="activate learnable equivalent transformation")     # ltl
    parser.add_argument("--lwc", action="store_true",
                        default=True,
                        help="activate learnable weight clipping")                # ltl
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_cluster", choices=["per_token", "per_tensor", "per_cluster"]) # ltl
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    # ltl新加的参数
    parser.add_argument("--a_dynamic", action="store_true", help="whether use dynamic quantization for act")
    parser.add_argument("--metric",type=str, default="ema_minmax", choices=["minmax", "ema_minmax"])
    parser.add_argument("--reorder", default="", help="for example: 012345")                                 
    # 0表示在omni之前要设置好各个activation的组大小。比如某个activation是768维，对应的clusters为32，那么每个cluster就要有24个channels            
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--weight_exp_quant", action="store_true",default=True,
                        help="use exponent quantization for weights") # ltl
    parser.add_argument("--w_symmetric", action="store_true",default=True)      # 经过我的测试，对指数量化来说，非对称量化竟然不如对称量化，loss也更大

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.calib_cache_dir:
        Path(args.calib_cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    if "opt" in args.net:
        args.model = f"facebook/{args.net}"
        if not os.path.exists(f"{args.model_cache_dir}/{args.net.split('-')[0]}"):
            os.makedirs(f"{args.model_cache_dir}/{args.net.split('-')[0]}/")
        args.model_cache_dir = (
            f"{args.model_cache_dir}/{args.net.split('-')[0]}/{args.net.split('-')[1]}"
        )
        cache_file = f"{args.model_cache_dir}/torch_model.pth"
        if os.path.exists(cache_file):
            lm = torch.load(cache_file)
        else:
            lm = OPTClass(args)
            torch.save(lm, cache_file)
        lm.model.eval()
    elif "llama-2" in args.net:
        args.model = f"llama/{args.net}"
        if not os.path.exists(f"{args.model_cache_dir}/{args.net.split('-')[0]}_2/"):
            os.makedirs(f"{args.model_cache_dir}/{args.net.split('-')[0]}/")
        args.model_cache_dir = (
            f"{args.model_cache_dir}/{args.net.split('-')[0]}_2/{args.net.split('-')[2]}"
        )
        cache_file = f"{args.model_cache_dir}/pytorch_model.pth"
        if os.path.exists(cache_file):
            lm = torch.load(cache_file)
        else:
            lm = LLAMAClass(args)
            torch.save(lm, cache_file)
        lm.model.eval()
    lm.seqlen = 2048
    for param in lm.model.parameters():
        param.requires_grad = False
    
    # 测试原始模型
    # evaluate(lm, args,logger)
    # return 

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        # "symmetric": args.symmetric if not args.weight_exp_quant else True,                                                 # 看来对权重用点对称量化问题不大，对激活值坚决不能用
        "symmetric": args.w_symmetric,
        "dynamic_method": args.w_dynamic_method,                                                                        # 对权重采用per-channel的量化吗
        "group_size": args.group_size,
        "lwc":args.lwc,                                                                                                 # 权重相比于激活值，有这么个参数
        "dynamic": True,                                                                                                # 权重的dynamic是true，只是为了对它进行per_channel的量化，不是真的动态
        "metric": "minmax"
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "dynamic": args.a_dynamic,
        "metric": args.metric,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "dynamic": args.a_dynamic,
        "metric": args.metric,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "dynamic": args.a_dynamic,
        "metric": args.metric,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "dynamic": args.a_dynamic,
        "metric": args.metric,
    }
    args.p_quant_params = {
        "n_bits": 16,                                                                       # 没有量化p
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'
    # args.act_scales = f'./ltl_scales/125m.pt'
    # args.act_shifts = f'./ltl_shifts/125m.pt'               # ltl

    # quantization
    # if args.wbits < 16 or args.abits <16:                     # ltl
    if args.wbits <= 16 or args.abits <=16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.calib_cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 
    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
