#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GVC_testmode=1

GPU_id=0

#scenes="apple backpack block creeper handwavy haru-sit mochi-high-five pillow space-out spin sriracha-tree teddy wheel"
scenes="apple"

# Training
for scene in $scenes; do

	output_dir=output/dycheck/${scene}
	mkdir -p ${output_dir}
	
	log_dir=${output_dir}/log_${scene}.txt
#	export CUDA_LAUNCH_BLOCKING=1
	#Third, downsample the point clouds generated in the second step.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dycheck/${scene}/colmap/dense/workspace/fused.ply data/dycheck/${scene}/points3D_downsample2.ply > ${log_dir}
	
	# Finally, train.
	CUDA_LAUNCH_BLOCKING=1 PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dycheck/${scene} --port 6020 --expname "dycheck/${scene}" --configs arguments/dycheck/default.py --GVC_testmode ${GVC_testmode} >> ${log_dir}
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/dycheck/${scene}"  --skip_train --configs arguments/dycheck/default.py --GVC_testmode ${GVC_testmode} >> ${log_dir}
	
	# evaluation
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/dycheck/${scene}" >> ${log_dir}
	
done


