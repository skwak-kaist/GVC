#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

# argument가 입력되면 dataset에 할당
if [ -z "$1" ]
  then
	scenes="apple block spin paper-windmill space-out teddy wheel" 
	echo "All scenes are used"
else
	scene_set=$1
	echo "Scenes: $scene_set"
	if [ $scene_set == 0 ]
	then 
		scenes="apple block"
		GPU_id=0
		port=1000
	elif [ $scene_set == 1 ]
	then 
		scenes="spin paper-windmill "
		GPU_id=1
		port=2000
	elif [ $scene_set == 2 ]
	then 
		scenes="space-out teddy"
		GPU_id=0
		port=3000
	elif [ $scene_set == 3 ]
	then 
		scenes="wheel"
		GPU_id=1
		port=4000
	elif [ $scene_set == "s1" ]
	then 
		scenes="apple block spin paper-windmill"
		GPU_id=0
		port=1000
	elif [ $scene_set == "s2" ]
	then 
		scenes="space-out teddy wheel"
		GPU_id=1
		port=2000
	elif [ $scene_set == "all" ]
	then 
		scenes="apple block spin paper-windmill space-out teddy wheel" 
		GPU_id=1
		port=1000
	elif [ $scene_set == "single" ]
	then
		scenes="paper-windmill" 
		GPU_id=0
		port=6700
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
	# port에 GVC_tetmode를 10을 곱해 더해줌
	port=$((port+GVC_testmode*100))
fi

# GVC dynamic mode
if [ -z "$3" ]
  then
	GVC_dynamic_mode=1
else
	GVC_dynamic_mode=$3
	# port에 GVC_tetmode를 더해줌
	port=$((port+GVC_dynamic_mode*10))
fi

# config number
if [ -z "$4" ]
  then
	config=config_v1
	test_version=gvc${GVC_testmode}.${GVC_dynamic_mode}.0
else
	config="config_v"$4
	test_version=gvc${GVC_testmode}.${GVC_dynamic_mode}.$4
	port=$((port+$4))
fi

GVC_Scale_Activation=1
GVC_Opacity_Activation=0


# output path
dataset=dycheck
output_path=${dataset}_${test_version}_${config}

cd ..

for scene in $scenes; do

	echo "scene: "$scene
	echo "port: "$port
	echo "GVC_testmode: "$GVC_testmode
	echo "GVC_dynamic_mode: "$GVC_dynamic_mode
	echo "config: "$config

	#Third, downsample the point clouds generated in the second step.
	echo "Downsampling the point cloud"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene}/colmap/dense/workspace/fused.ply data/${dataset}/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	echo "Training the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/${config}.py --GVC_testmode ${GVC_testmode} --GVC_Dynamics ${GVC_dynamic_mode} 
	
	# rendering
	echo "Rendering the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --configs arguments/${dataset}/${config}.py --GVC_testmode ${GVC_testmode} --GVC_Dynamics ${GVC_dynamic_mode} 

	# rendering canonical frame
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --skip_test --skip_video --configs arguments/${dataset}/${config}.py --canonical_frame_render --GVC_testmode ${GVC_testmode} 
	
	# evaluation
	echo "Evaluating the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}"

done

python collect_metric.py --output_path "output/${output_path}" 




