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
	GPU_id=0
	port=5000
else
	scenes=$scene_set
	GPU_id=1
	port=5000
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

dataset=dycheck
test_version=GVC_v$config_number
output_path=${date}_${dataset}_${test_version}

cd ..

mkdir output/${output_path}

# training time 측정을 위한 txt 파일 생성
echo "training time" > "output/${output_path}/training_time.txt"

echo "training log" > "output/${output_path}/training_log.txt"

for scene in $scenes; do

	echo "########################################"
	echo "scene: "$scene
	echo "config: "$config
	echo "GPU" $GPU_id
	echo "########################################"


	#Third, downsample the point clouds generated in the second step.
	#echo "Downsampling the point cloud"
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene}/colmap/dense/workspace/fused.ply data/${dataset}/${scene}/points3D_downsample2.ply

	# 만약 $train이 1이 아니면 skip
	if [ $train == 1 ]
	then
		# Finally, train.
		# 현재 시간 기록
		start_time=$(date '+%s')

		echo "Training the model"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/${config}.py >> "output/${output_path}/training_log.txt"

		# 학습시간 기록
		end_time=$(date '+%s')
		diff=$((end_time - start_time))
		hour=$((diff / 3600 % 24))
		echo "Training time: $(($diff / 3600)) hours, $(($diff % 3600 / 60)) minutes, $(($diff % 60)) seconds" >> "output/${output_path}/training_time.txt"

	else
		echo "Skip training"
	fi

	# rendering
	echo "Rendering the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --configs arguments/${dataset}/${config}.py >> "output/${output_path}/training_log.txt"

	# rendering canonical frame
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --skip_test --skip_video --configs arguments/${dataset}/${config}.py --canonical_frame_render
	
	# evaluation
	echo "Evaluating the model"
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" >> "output/${output_path}/training_log.txt"

done

python collect_metric.py --output_path "output/${output_path}" 




