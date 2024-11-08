#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

# argument가 입력되면 dataset에 할당
if [ -z "$1" ]
  then
	scenes="lame_salmon_1 flame_steak sear_steak coffee_martini cook_spinach cut_roasted_beef" 
	echo "All scenes are used"
else
	scene_set=$1
	echo "Scenes: $scene_set"
	if [ $scene_set == "s1" ]
	then 
		scenes="flame_salmon_1 flame_steak sear_steak "
		GPU_id=0
		port=6020
	elif [ $scene_set == "s2" ]
	then 
		scenes="coffee_martini cook_spinach cut_roasted_beef"
		GPU_id=1
		port=6021
	elif [ $scene_set == "all" ]
	then 
		scenes="flame_salmon_1 flame_steak sear_steak coffee_martini cook_spinach cut_roasted_beef" 
		GPU_id=1
		port=6022
	else
		echo "Invalid scene set"
		exit 1
	fi
fi

# GVC parameters
if [ -z "$2" ]
  then
	GVC_testmode=2
else
	GVC_testmode=$2
	# port에 GVC_tetmode를 더해줌
	port=$((port+GVC_testmode))
fi

# config number
if [ -z "$3" ]
  then
	config=config_v1
	test_version=gvc${GVC_testmode}.0
else
	config="config_v"$3
	test_version=gvc${GVC_testmode}.$3
	port=$((port+$3))
fi

GVC_Scale_Activation=1
GVC_Opacity_Activation=0


# output path
dataset=dynerf
output_path=${dataset}_${test_version}_${config}

cd ..

for scene in $scenes; do

	echo "scene: "$scene

	#Third, downsample the point clouds generated in the second step.
	echo "Downsampling the point cloud"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene}/colmap/dense/workspace/fused.ply data/${dataset}/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	echo "Training the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/${config}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering
	echo "Rendering the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --configs arguments/${dataset}/${config}.py --GVC_testmode ${GVC_testmode} 

	# rendering canonical frame
	# PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --skip_test --skip_video --configs arguments/${dataset}/${config}.py --canonical_frame_render --GVC_testmode ${GVC_testmode} 
	
	# evaluation
	echo "Evaluating the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}"

done

python collect_metric.py --output_path "output/${output_path}" --dataset dynerf




