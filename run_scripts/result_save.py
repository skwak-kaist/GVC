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

def save_result(output_folder, save_path, dataset):
    print(f'Saving dycheck result from {output_folder} to {save_path}')
    
    # output_folder 안의 폴더 리스트
    if dataset == 'dycheck':
        folder_list = ["apple", "block", "spin", "paper-windmill", "space-out", "teddy", "wheel"] 
    elif dataset == 'hypernerf':
        folder_list = ["aleks-teapot", "chickchicken", "cut-lemon1", "hand1", "slice-banana", "torchocolate", 
                       "americano", "cross-hands1", "espresso", "keyboard", "oven-mitts", "split-cookie", "tamping", 
                       "3dprinter", "broom", "chicken", "peel-banana"]
    
    for folder in folder_list:
        subfolder_path = os.path.join('../output', output_folder, folder)
        
        # subfolder_path 가 존재하지 않으면 다음 폴더로 넘어감
        if not os.path.exists(subfolder_path):
            continue
        
        print(f'Processing {subfolder_path}')
        
        # save_path/output_folder/folder 경로가 있는지 검사
        folder_path = os.path.join(save_path, output_folder, folder)
        
        # 폴더가 없으면 생성하고 있으면 넘어감
        if not os.path.exists(folder_path):
            os.makedirs(os.path.join(save_path, output_folder, folder), exist_ok=True)
        else:
            print(f'{folder_path} already exists. Skipping...')
            continue
        
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
        output_path_p = os.path.join(save_path, output_folder, folder, 'point_cloud')
        
        #output path가 없으면 생성
        os.makedirs(output_path_p, exist_ok=True)
        
        os.system(f'cp -r {os.path.join(point_cloud_path, point_cloud_folder)} {output_path_p}')
        
        # test 결과 복사
        test_path = os.path.join(subfolder_path, 'test')
        # test 경로가 존재하지 않으면 다음 폴더로 넘어감
        if not os.path.exists(test_path):
            continue
        # test 경로 아래의 폴더 리스트를 정렬하고 가장 마지막 폴더를 가져옴
        test_folder = sorted(os.listdir(test_path))[-1]
        
        output_path_t = os.path.join(save_path, output_folder, folder, 'test')
        os.makedirs(output_path_t, exist_ok=True)
        # test 폴더 복사
        os.system(f'cp -r {os.path.join(test_path, test_folder)} {output_path_t}')
        
        # video 결과 복사
        '''
        video_path = os.path.join(subfolder_path, 'video')
        # video 경로가 조재하지 않으면 다음 폴더로 넘어감
        
        # video 경로 아래의 폴더 리스트를 정렬하고 가장 마지막 폴더를 가져옴
        video_folder = sorted(os.listdir(video_path))[-1]
        output_path_v = os.path.join(save_path, output_folder, folder, 'video')
        os.makedirs(output_path_v, exist_ok=True)
        
        # video 폴더 복사
        os.system(f'cp -r {os.path.join(video_path, video_folder)} {output_path_v}')
    	'''
    


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

        # save_result 함수 호출
        save_result(output_folder, args.save_path, "dycheck")
        save_result(output_folder, args.save_path, "hypernerf")
