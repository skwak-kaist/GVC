#!/bin/bash 
GPU_id=1
scenes="cut_roasted_beef cook_spinach flame_salmon flame_steak sear_steak " # coffee_martini
#scenes="cook_spinach" 

for scene in $scenes; do

	# First, extract the frames of each video.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/preprocess_dynerf.py --datadir data/dynerf/${scene}

	# Second, generate point clouds from input data.
	bash colmap.sh data/dynerf/${scene} llff

	#Third, downsample the point clouds generated in the second step.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dynerf/${scene}/colmap/dense/workspace/fused.ply data/dynerf/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dynerf/${scene} --port 6018 --expname "dynerf/${scene}" --configs arguments/dynerf/${scene}.py 
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/dynerf/${scene}"  --skip_train --configs arguments/dynerf/${scene}.py 
	
	# evaluation
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/dynerf/${scene}" 

done 



