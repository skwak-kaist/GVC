#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GPU_id=0
GVC_testmode=1
port=6030

scenes="cut_roasted_beef cook_spinach flame_salmon_1 flame_steak sear_steak coffee_martini" 
#scenes="cut_roasted_beef" 

output_path=dynerf_anchor2

for scene in $scenes; do

	#export CUDA_LAUNCH_BLOCKING=1
	
	#Third, downsample the point clouds generated in the second step.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dynerf/${scene}/colmap/dense/workspace/fused.ply data/dynerf/${scene}/points3D_downsample2.ply 
	
	# Finally, train.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dynerf/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering canonical frame
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --skip_test --skip_video --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} --canonical_frame_render
	
	# evaluation
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" 
	

	
done

PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python collect_metric.py --output_path "output/${output_path}" 
