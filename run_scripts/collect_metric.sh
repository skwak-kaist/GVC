#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate GVC

folder_name=$1

output_path=${folder_name}

dataset="dycheck"

cd ..

python collect_metric.py --output_path "output/${output_path}" --dataset ${dataset}
	 
 



