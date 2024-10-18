#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GVC_testmode=1
port=6000
GPU_id=0

scenes="flame_salmon_1 flame_steak sear_steak coffee_martini cook_spinach cut_roasted_beef" 
#scenes="cut_roasted_beef" 

# Training
for scene in $scenes; do

	output_dir=output/dynerf/${scene}
	mkdir -p ${output_dir}
	
	log_dir=${output_dir}/log_${scene}.txt
	#export CUDA_LAUNCH_BLOCKING=1
	#Third, downsample the point clouds generated in the second step.
	
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dynerf/${scene}/colmap/dense/workspace/fused.ply data/dynerf/${scene}/points3D_downsample2.ply > ${log_dir}
	
	# Finally, train.
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dynerf/${scene} --port 6022 --expname "dynerf/${scene}" --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} >> ${log_dir}
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/dynerf/${scene}"  --skip_train --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} 
	
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/dynerf/${scene}"  --skip_train --skip_test --skip_video --configs arguments/dynerf/${scene}.py --GVC_testmode ${GVC_testmode} --canonical_frame_render
	
	#PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/dynerf/${scene}" >> ${log_dir}
	
done

