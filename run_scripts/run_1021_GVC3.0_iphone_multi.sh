#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

# 파일명을 읽어옴
date=$(basename $0 | cut -d'_' -f2)
echo "date: "$date

# 1st argument: dataset
scene_set=$1
if [ $scene_set == 0 ]
then 
	scenes="apple block"
	GPU_id=0
	port=5100
elif [ $scene_set == 1 ]
then 
	scenes="spin paper-windmill "
	GPU_id=1
	port=5200
elif [ $scene_set == 2 ]
then 
	scenes="space-out teddy"
	GPU_id=0
	port=5300
elif [ $scene_set == 3 ]
then 
	scenes="wheel"
	GPU_id=1
	port=5400
elif [ $scene_set == "s1" ]
then 
	scenes="apple block spin paper-windmill"
	GPU_id=0
	port=5100
elif [ $scene_set == "s2" ]
then 
	scenes="space-out teddy wheel"
	GPU_id=1
	port=5200
elif [ $scene_set == "all" ]
then 
	scenes="apple block spin paper-windmill space-out teddy wheel" 
	GPU_id=1
	port=5000
else
	scenes=$scene_set
fi

echo "This scripts runs for scenes: $scenes"

# 2nd arguments: GVC parameters
GVC_testmode=$2
echo "GVC_testmode: "$GVC_testmode

# 3rd arguments: GVC dynamic mode
GVC_dynamic_mode=$3
echo "GVC_dynamic_mode: "$GVC_dynamic_mode

# 4th arguments: config number
config=$4

# 5th arguments: port
if [ -z "$5" ]
  then
	echo "port: "$port
else
	port=$5
	echo "port: "$port
fi

# 7th arguments: train or not
if [ -z "$6" ]
  then
	train=1
else
	train=$6
fi

dataset=dycheck
test_version=gvc${GVC_testmode}.${GVC_dynamic_mode}.$config
output_path=${dataset}_${test_version}_${date}

cd ..

for scene in $scenes; do

	echo "########################################"
	echo "scene: "$scene
	echo "GVC_testmode: "$GVC_testmode
	echo "GVC_dynamic_mode: "$GVC_dynamic_mode
	echo "config: "$config
	echo "########################################"


	#Third, downsample the point clouds generated in the second step.
	#echo "Downsampling the point cloud"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene}/colmap/dense/workspace/fused.ply data/${dataset}/${scene}/points3D_downsample2.ply
	
	# 만약 $train이 1이 아니면 skip
	if [ $train == 1 ]
	then
		# Finally, train.
		echo "Training the model"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/${config}.py --GVC_testmode ${GVC_testmode} --GVC_Dynamics ${GVC_dynamic_mode} 
	else
		echo "Skip training"
	fi

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




