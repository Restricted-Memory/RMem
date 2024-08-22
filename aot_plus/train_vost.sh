exp="aotplus"
# exp="debug"
gpu_num="4"
devices="1,2,3,4"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
model="r50_deaotl"
# model="swinb_aotl"
	
stage="pre_vost"
stage="pre_vost_2"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"
CUDA_VISIBLE_DEVICES=${devices} python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--batch_size 8 \
	--fix_random \
	# --log ./debug_logs \
	# --debug_fix_random


dataset="vost"
# dataset="long_videos"
split="val"
eval_name="debug"
# eval_name="max_mem_1_7_drop_layer_mean_0012_ucb_reset_0_0"
# eval_name="max_mem_1_7_drop_layer_mean_0012_ucb_reset_0_0_plus_4_mul_0.15"
# eval_name="max_mem_1_7_drop_layer_mean_0012_ucb_reset_0_8_plus_8_mul_0.11"
# eval_name="max_mem_1_7_drop_layer_mean_0012_mov_mean_0.8_ucb_reset_0_8_plus_8_mul_0.1"
# eval_name="max_mem_1_7_drop_layer_mean_0012_mov_mean_0.8_ucb_reset_0_8_plus_28_mul_0.1"
# eval_name="max_mem_1_7_drop_layer_mean_0012_focus_mov_mean_0.8"
# eval_name="max_mem_1_8_drop_layer_0_focus"
# eval_name="max_mem_1_8_drop_layer_0_focus_mov_mean_0.5_ucb_reset_0_plus_8_mul_0.2"
# eval_name="max_mem_1_8_drop_layer_mean_012_focus_mov_mean_0.9"
# eval_name="max_mem_1_8_drop_layer_mean_012_focus_mov_mean_0.9_ucb_reset_0_plus_8_mul_0.01"
# eval_name="gap_8_max_mem_1_7_drop_layer_mean_0012_moving_mean"
# eval_name="max_gap_8"
# eval_name="max_mem_1_19"
# eval_name="max_mem_8"
# eval_name="max_mem_2_6"
# eval_name="max_mem_1_7"
# eval_name="max_mem_1_8_nearest"
# eval_name="max_mem_1_7_nearest_flip"
# eval_name="max_mem_1_7_nearest_flip_drop_layer_012_focus_mov_mean_0.8"
# eval_name="max_mem_1_8_nearest_flip_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1000"
# eval_name="max_mem_1_7_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1.4"
# eval_name="max_mem_1_8_nearest_flip"
# eval_name="max_mem_1_8_nearest_flip_drop_layer_0_focus_mov_mean_0.8"
# eval_name="max_mem_1_8_nearest_flip_drop_layer_012_focus_mov_mean_0.9"
eval_name="max_mem_1_8_nearest_flip_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1.5"
# eval_name="max_mem_1_8_nearest_exact_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1.5"
# eval_name="max_mem_1_8_nearest_exact"
# eval_name="max_mem_1_7_nearest_flip"
# eval_name="495_366_max_mem_1_10"
# eval_name="494_374_max_mem_1_11"
# eval_name="max_mem_1_7_rand_11"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path "${result_path}" \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random \
	# --debug_fix_random

result_path="${result_path}/eval/${dataset}/${eval_name}/"
echo "result_path=$result_path"


model_name=$(python -c "from configs.models.$model import ModelConfig ;print(ModelConfig().MODEL_NAME)")
cd ../evaluation
python ./evaluation_method.py --results_path "../aot_plus/${result_path}" --dataset_path ${dataset} --re