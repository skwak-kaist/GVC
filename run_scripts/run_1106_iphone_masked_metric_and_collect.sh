#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate maskmetric

# "output" folder안에 있는 모든 directory 중에 폴더 명에 dycheck가 들어간 폴더를 찾아서 list로 만듦

cd ..

output_path="output"
scenes="apple block spin paper-windmill space-out teddy wheel"
#folder_list="1030_dycheck_GVC_v5.0.8.0.0.t1-2 1030_dycheck_GVC_v5.0.8.0.0.t1 1030_dycheck_GVC_v5.0.8.0.0.t2 1030_dycheck_GVC_v5.0.8.0.0.t2-2 1030_dycheck_GVC_v5.0.8.0.0.t3 1030_dycheck_GVC_v5.0.8.0.0.t3-2 1030_dycheck_GVC_v5.0.8.0.0.t4 1030_dycheck_GVC_v5.0.8.0.0.t4-2"

#folder_list="1030_dycheck_GVC_v5.0.8.0.0.t1-2"

for dir in $(ls -d ${output_path}/*dycheck*); do
#for dir in $folder_list; do
    echo $dir
    for scene in $scenes; do
        echo $scene
        CUDA_VISIBLE_DEVICES=1 python metrics_masked.py --model_path ${dir}/${scene} --data_path data/dycheck/${scene} --lpips_only 1
        #CUDA_VISIBLE_DEVICES=1 python metrics_masked.py --model_path output/${dir}/${scene} --data_path data/dycheck/${scene} --lpips_only 1
    done
    
done

for dir in $(ls -d ${output_path}/*dycheck*); do
    echo $dir
    python collect_metric.py --output_path ${dir} --dataset dycheck --mask 1 --lpips_only 1
done


