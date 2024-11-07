#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate maskmetric

# "output" folder안에 있는 모든 directory 중에 폴더 명에 dycheck가 들어간 폴더를 찾아서 list로 만듦

cd ..

output_path="output"
scenes="apple block spin paper-windmill space-out teddy wheel"


for dir in $(ls -d ${output_path}/*dycheck*); do
    echo $dir
    python collect_metric.py --output_path ${dir} --dataset dycheck --mask 1
done


