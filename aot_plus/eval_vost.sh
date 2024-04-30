exp="aotplus"
# exp="debug"
gpu_num="3"
devices="0,1,2,3,4,5,6,7"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
# model="r50_aotl_LSTT_pullback"
# model="r50_topdown_aotl"
# model="swinb_aotl"
	
stage="pre_vost"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"


dataset="vost"
split="val"
eval_name="debug"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path ${result_path} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--ckpt_path pretrain_models/aotplus_R50_AOTL_Temp_pe_Slot_4_ema_20000.pth \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random \
	# --debug_fix_random


result_path="${result_path}/eval/${dataset}/${eval_name}/"
echo "result_path=$result_path"


model_name=$(python -c "from configs.models.$model import ModelConfig ;print(ModelConfig().MODEL_NAME)")
cd ../evaluation
python ./evaluation_method.py --results_path "../aot_plus/${result_path}" --dataset_path ${dataset} --re