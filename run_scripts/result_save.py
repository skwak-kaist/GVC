#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser

# ../output 안에 있는 모든 폴더
def get_all_output_folders():
    return [f for f in os.listdir('../output') if os.path.isdir(os.path.join('../output', f))]

# 각 output folder의 type을 구분
# 폴더명에 dycheck이 들어가면 dycheck, hypernerf가 들어가면 hypernerf
def get_output_folder_type(output_folder):
    if 'dycheck' in output_folder:
        return 'dycheck'
    elif 'hypernerf' in output_folder:
        return 'hypernerf'
    else:
        return 'unknown'

def save_result_dycheck(output_folder, save_path):
    print(f'Saving dycheck result from {output_folder} to {save_path}')
    
    # output_folder 안의 폴더 리스트
    folder_list = ["apple", "block", "spin", "paper-windmill", "space-out", "teddy", "wheel"] 
    
    for folder in folder_list:
        subfolder_path = os.path.join('../output', output_folder, folder)
        
        # subfolder_path 가 존재하지 않으면 다음 폴더로 넘어감
        if not os.path.exists(subfolder_path):
            continue
        
        print(f'Processing {subfolder_path}')
        
        # save_path/output_folder/folder 경로 생성
        os.makedirs(os.path.join(save_path, output_folder, folder), exist_ok=True)
        
        # cfg 파일 복사
        os.system(f'cp {subfolder_path}/cfg_args {os.path.join(save_path, output_folder, folder)}')

        # point_cloud 경로
        point_cloud_path = os.path.join(subfolder_path, 'point_cloud')
        # point_cloud가 존재하지 않으면 다음 폴더로 넘어감
        if not os.path.exists(point_cloud_path):
            continue
        # point_cloud_path의 폴더 리스트를 정렬하고 가장 마지막 폴더를 가져옴
        point_cloud_folder = sorted(os.listdir(point_cloud_path))[-1]
        # point_cloud 폴더 복사
        os.system(f'cp -r {os.path.join(point_cloud_path, point_cloud_folder)} {os.path.join(save_path, output_folder, 'point_cloud', folder)}')
        
        # test 결과 복사
        test_path = os.path.join(subfolder_path, 'test')
        # test 경로가 존재하지 않으면 다음 폴더로 넘어감
        if not os.path.exists(test_path):
            continue
        # test 경로 아래의 폴더 리스트를 정렬하고 가장 마지막 폴더를 가져옴
        test_folder = sorted(os.listdir(test_path))[-1]
        # test 폴더 복사
        os.system(f'cp -r {os.path.join(test_path, test_folder)} {os.path.join(save_path, output_folder, 'test', folder)}')
        
        # video 결과 복사
        video_path = os.path.join(subfolder_path, 'video')
        # video 경로가 조재하지 않으면 다음 폴더로 넘어감
        
        # video 경로 아래의 폴더 리스트를 정렬하고 가장 마지막 폴더를 가져옴
        video_folder = sorted(os.listdir(video_path))[-1]
        # video 폴더 복사
        os.system(f'cp -r {os.path.join(video_path, video_folder)} {os.path.join(save_path, output_folder, 'video', folder)}')
    
    


# main 함수
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="../output/PC1")
    args = parser.parse_args()
    
    
    # get_all_output_folders 함수 호출
    output_folders = get_all_output_folders()
    
    for output_folder in output_folders:
        # get_output_folder_type 함수 호출
        output_folder_type = get_output_folder_type(output_folder)
        print(f'Output folder: {output_folder}, type: {output_folder_type}')

        if output_folder_type == 'dycheck':
            print('saving dycheck')
            save_result_dycheck(output_folder, args.save_path)
    