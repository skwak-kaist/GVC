#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

# 파일명을 읽어옴
date=$(basename $0 | cut -d'_' -f2)
echo "date: "$date

# 1st argument: dataset
scene_set=$1
	echo "Scenes: $scene_set"
	if [ $scene_set == 0 ]
	then 
		data_subset="interp"
		scenes="aleks-teapot chickchicken"
		scnen_paths="interp/interp_aleks-teapot/aleks-teapot \
		interp/interp_chickchicken/chickchicken"
		
		GPU_id=0
		port=6020

	elif [ $scene_set == 1 ]
	then 
		data_subset="interp"
		scenes="cut-lemon1 hand1"
		scnen_paths="interp/interp_cut-lemon/cut-lemon1 \
		interp/interp_hand/hand1-dense-v2"
		
		
		GPU_id=1
		port=6021
	elif [ $scene_set == 2 ]
	then 
		data_subset="interp"
		scenes="slice-banana torchocolate"
		scnen_paths="interp/interp_slice-banana/slice-banana \
		interp/interp_torchocolate/torchocolate"
		

		GPU_id=0
		port=6022
		
	elif [ $scene_set == 3 ]
	then 
		data_subset="misc"
		scenes="americano oven-mitts"
		scnen_paths="misc/misc_americano/americano \
		misc/misc_oven-mitts/oven-mitts"

		GPU_id=1
		port=6023
	
	elif [ $scene_set == s0 ]
	then 
		data_subset="misc"
		scenes="americano cross-hands1 espresso keyboard oven-mitts"
		scnen_paths="misc/misc_americano/americano \
		misc/misc_cross-hands/cross-hands1 \
		misc/misc_espresso/espresso \
		misc/misc_keyboard/keyboard \
		misc/misc_oven-mitts/oven-mitts"
		
		GPU_id=0
		port=6020
	
	elif [ $scene_set == s1 ]
	then 
		data_subset="vrig"
		scenes="3dprinter broom chicken peel-banana split-cookie tamping"
		scnen_paths="vrig/vrig_3dprinter/vrig-3dprinter \
		vrig/vrig_broom/broom2 \
		vrig/vrig_chicken/vrig-chicken \
		vrig/vrig_peel-banana/vrig-peel-banana \
		misc/misc_split-cookie/split-cookie \
		misc/misc_tamping/tamping"
		
		GPU_id=1
		port=6021
	
	elif [ $scene_set == r0 ]
	then 
		data_subset="misc"
		scenes="americano oven-mitts"
		scnen_paths="misc/misc_americano/americano \
		misc/misc_oven-mitts/oven-mitts"
		
		GPU_id=0
		port=6020
	
	elif [ $scene_set == r1 ]
	then 
		data_subset="vrig"
		scenes="split-cookie broom"
		scnen_paths="misc/misc_split-cookie/split-cookie \
		vrig/vrig_broom/broom2"
		
		GPU_id=1
		port=6021
	
		
	elif [ $scene_set == "all" ]
	then 
		data_subset="all"
		scenes="aleks-teapot chickchicken cut-lemon1 hand1 slice-banana torchocolate \
		americano cross-hands1 espresso keyboard oven-mitts split-cookie tamping \
		3dprinter broom chicken peel-banana"
		scnen_paths="interp/interp_aleks-teapot/aleks-teapot \
		interp/interp_chickchicken/chickchicken \
		interp/interp_cut-lemon/cut-lemon1 \
		interp/interp_hand/hand1-dense-v2 \
		interp/interp_slice-banana/slice-banana \
		interp/interp_torchocolate/torchocolate \
		misc/misc_americano/americano \
		misc/misc_cross-hands/cross-hands1 \
		misc/misc_espresso/espresso \
		misc/misc_keyboard/keyboard \
		misc/misc_oven-mitts/oven-mitts \
		misc/misc_split-cookie/split-cookie \
		misc/misc_tamping/tamping \
		vrig/vrig_3dprinter/vrig-3dprinter \
		vrig/vrig_broom/broom2 \
		vrig/vrig_chicken/vrig-chicken \
		vrig/vrig_peel-banana/vrig-peel-banana"
		
		GPU_id=1
		port=6020
	else
	
		echo "Invalid scene set"
		exit 1
	fi


echo "This scripts runs for scenes: $scenes"

# 2nd argument: config_number
config_number=$2
config="config_v"$config_number

# 3rd argument: port
if [ -z "$3" ]
  then
	echo "port: "$port
else
	port=$3
	echo "port: "$port
fi

# 4th arguments: gpu
if [ -z "$4" ]
  then
	GPU_id=${GPU_id}
else
	GPU_id=$4
fi

# 5th arguments: train or not
if [ -z "$5" ]
  then
	train=1
else
	train=$5
fi

dataset=hypernerf
test_version=GVC_v$config_number
output_path=${date}_${dataset}_${test_version}

cd ..

mkdir output/${output_path}

# training time 측정을 위한 txt 파일 생성
echo "training time" > "output/${output_path}/training_time.txt"

# training log
echo "training log" > "output/${output_path}/training_log.txt"

for scene in $scenes; do

	scene_path=$(echo $scnen_paths | cut -d' ' -f$((idx+1)))

	echo "########################################"
	echo "scene: "$scene
	echo "scene path: "$scene_path
	echo "config: "$config
	echo "GPU" $GPU_id
	echo "########################################"

	#bash colmap.sh data/${dataset}/${scene_path} ${dataset}


	# rendering
	echo "Rendering the model"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --configs arguments/${dataset}/${config}.py >> "output/${output_path}/training_log.txt"

	# rendering canonical frame
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --skip_test --skip_video --configs arguments/${dataset}/${config}.py --canonical_frame_render
	
	# evaluation
	#echo "Evaluating the model"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" >> "output/${output_path}/training_log.txt"

	# idx +1
	idx=$(($idx + 1))

done

#python collect_metric.py --output_path "output/${output_path}" --dataset ${dataset}




