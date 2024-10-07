#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

GPU_id=1
GVC_testmode=1
port=6028

config=anchor_v0

output_paths="dycheck_anchor_v0 dycheck_anchor_v5 dycheck_anchor_v6 dycheck_anchor_v7 dycheck_anchor_v8"

cd ..

for output_path in $output_paths; do

	python collect_metric.py --output_path "output/${output_path}"
	 
done 



