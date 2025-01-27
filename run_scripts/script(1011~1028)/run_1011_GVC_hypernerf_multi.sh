#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

# argument가 입력되면 dataset에 할당
if [ -z "$1" ]
  then
	scenes="apple backpack block creeper handwavy haru-sit mochi-high-five pillow space-out spin sriracha-tree teddy wheel" 
	echo "All scenes are used"
else
	scene_set=$1
	echo "Scenes: $scene_set"
	if [ $scene_set == 0 ]
	then 
		scenes="apple backpack block"
		GPU_id=0
		port=6020
	elif [ $scene_set == 1 ]
	then 
		scenes="creeper handwavy haru-sit"
		GPU_id=1
		port=6021
	elif [ $scene_set == 2 ]
	then 
		scenes="mochi-high-five pillow space-out"
		GPU_id=0
		port=6022
	elif [ $scene_set == 3 ]
	then 
		scenes="spin sriracha-tree teddy wheel"
		GPU_id=1
		port=6023
	elif [ $scene_set == "all" ]
	then 
		scenes="apple backpack block creeper handwavy haru-sit mochi-high-five pillow space-out spin sriracha-tree teddy wheel" 
		GPU_id=1
		port=6020
	else
		echo "Invalid scene set"
		exit 1
	fi
fi

# GVC parameters
GVC_testmode=1
GVC_Scale_Activation=1
GVC_Opacity_Activation=0

# Test versions
test_version=gvc1.2
config=anchor_v9

# output path
output_path=dycheck_${test_version}_${config}

cd ..

for scene in $scenes; do

	echo "scene: "$scene

	#Third, downsample the point clouds generated in the second step.
	echo "Downsampling the point cloud"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dycheck/${scene}/colmap/dense/workspace/fused.ply data/dycheck/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	echo "Training the model"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dycheck/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/dycheck/default_${config}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering
	echo "Rendering the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --configs arguments/dycheck/default_${config}.py --GVC_testmode ${GVC_testmode} 

	# rendering canonical frame
	# PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --skip_test --skip_video --configs arguments/dycheck/default_${config}.py --canonical_frame_render --GVC_testmode ${GVC_testmode} 
	
	# evaluation
	echo "Evaluating the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" 
done 

python collect_metric.py --output_path "output/${output_path}" 

