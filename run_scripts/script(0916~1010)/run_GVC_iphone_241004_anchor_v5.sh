#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GPU_id=1
GVC_testmode=1
port=6020

scenes="apple backpack block creeper handwavy haru-sit mochi-high-five pillow space-out spin sriracha-tree teddy wheel" 
#scenes="flame_salmon_1" 

config=anchor_v5
output_path=dycheck_${config}

cd ..

for scene in $scenes; do

	#Third, downsample the point clouds generated in the second step.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/dycheck/${scene}/colmap/dense/workspace/fused.ply data/dycheck/${scene}/points3D_downsample2.ply
	
	# Finally, train.
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/dycheck/${scene} --port ${port} --expname "${output_path}/${scene}" --configs arguments/dycheck/default_${config}.py --GVC_testmode ${GVC_testmode} 
	
	# rendering
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --configs arguments/dycheck/default_${config}.py --GVC_testmode ${GVC_testmode} 
	

	# rendering canonical frame
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}"  --skip_train --skip_test --skip_video --configs arguments/dycheck/default_${config}.py --canonical_frame_render --GVC_testmode ${GVC_testmode} 
	
	# evaluation
	PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" 
done 

PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python collect_metric.py --output_path "output/${output_path}" 

