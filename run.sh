CUDA_VISIBLE_DEVICES=0 python main.py --model ../LLMs/opt/125m/models—facebook—opt-125m  --epochs 20 --output_dir ./log/opt-125m-w3a16 --eval_ppl --wbits 3 --abits 16 --lwc --net opt-125m



# opt-125m部分
CUDA_VISIBLE_DEVICES=7 python main.py --net opt-125m --epochs 20 --wbits 3 --abits 16 --lwc --let
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --net opt-125m --epochs 20 --wbits 3 --abits 16 --group_size 128 --output_dir ./log/opt-125m-w3a16-g128 --lwc --let 

# opt-1.3b部分
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --net opt-1.3b --epochs 20 --wbits 3 --abits 16 --output_dir ./log/opt-1.3b-w3a16 --lwc --let 
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --net opt-1.3b --epochs 20 --wbits 3 --abits 16 --group_size 128 --output_dir ./log/opt-1.3b-w3a16-g128 --lwc --let 

# 修改后的代码运行
# opt-125m
# 对激活值做per_token的动态量化
CUDA_VISIBLE_DEVICES=7 python main.py --net opt-125m --epochs 20 --wbits 3 --abits 8 --lwc --let --a_dynamic --a_dynamic_method per_token
# 对激活值做per_tensor的静态量化
CUDA_VISIBLE_DEVICES=7 python main.py --net opt-125m --epochs 20 --wbits 3 --abits 8 --lwc --let --a_dynamic_method per_tensor
# 对激活值做per_cluster的静态量化，但是没有reorder
CUDA_VISIBLE_DEVICES=3 python main.py --net opt-125m --epochs 20 --wbits 3 --abits 8 --lwc --let --a_dynamic_method per_cluster --reorder 0 




CUDA_VISIBLE_DEVICES=6 python main.py --net opt-6.7b --epochs 20 --wbits 4 --abits 4 --lwc --let --a_dynamic --a_dynamic_method per_token



CUDA_VISIBLE_DEVICES=6 python main.py --epochs 20 --wbits 3 --abits 6 --R1_clusters 32 --R2_clusters 4 --R3_clusters 4 --R4_clusters 32 --R5_clusters 32 --a_dynamic_method per_cluster


CUDA_VISIBLE_DEVICES=7 python main.py --epochs 0 --wbits 4 --abits 16 --reorder ""