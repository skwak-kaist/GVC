#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GPU_id=1
GVC_testmode=1
port=6027

scenes="flame_salmon_1 flame_steak sear_steak coffee_martini cook_spinach cut_roasted_beef" 

config=anchor_v0
dataset=dynerf
output_path=${dataset}_${config}

cd ..

for scene in $scenes; do

	#Third, downsample the point clouds generated in the second step.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene}/colmap/dense/workspace/fused.ply data/${dataset}/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/default_${config}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --configs arguments/${dataset}/default_${config}.py --GVC_testmode ${GVC_testmode} 
	

	# rendering canonical frame
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --skip_test --skip_video --configs arguments/${dataset}/default_${config}.py --canonical_frame_render --GVC_testmode ${GVC_testmode} 
	
	# evaluation
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" 
done 

PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python collect_metric.py --output_path "output/${output_path}" --dataset dynerf

