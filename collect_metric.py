
import os, sys
import json
from argparse import ArgumentParser, Namespace

def get_folder_list(dataset):
    if dataset == "dycheck":
        # apple, backpack, block, creeper, handwavy, haru-sit, mochi-high-five, pillow, space-out, spin, sriracha-tree, teddy, wheel
        # folder_list = ["apple", "backpack", "block", "creeper", "handwavy", "haru-sit", "mochi-high-five", "pillow", "space-out", "spin", "sriracha-tree", "teddy", "wheel"]
        # apple, block, paper-windmill, space-out, spin, teddy, wheel
        folder_list = ["apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel"]
        
    elif dataset == "dynerf":
        # coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
        folder_list = ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"]
        
    return folder_list


def collect_metric(folder_list, output_path):
    
    print(output_path)
    
    #print(folder_list)

    psnr_results = {}
    ssim_results = {}
    msssim_results = {}
    lpips_vgg_results = {}
    lpips_alex_results = {}    
    total_results = {}

    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        
        # json 파일이 없는 경우 pass
        if not os.path.exists(json_path):
            continue
        
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]
                
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['SSIM']
        msssim_results[folder] = results[result_key]['MS-SSIM']
        lpips_vgg_results[folder] = results[result_key]['LPIPS-vgg']
        lpips_alex_results[folder] = results[result_key]['LPIPS-alex']
        
        total_results[folder] = results[result_key]

        # print psnr result
        print(f"{folder} : {results[result_key]['PSNR']}")
    
    #print(total_results)
    # json으로 저장
    
    # output path의 가장 마지막 폴더 이름
    output_folder_name = output_path.split("/")[-1]
    
    with open(os.path.join(output_path, output_folder_name+ "_total_results.json"), 'w') as f:
        json.dump(total_results, f)

    # txt 파일로 저장
    with open(os.path.join(output_path, output_folder_name+ "_total_results.txt"), 'w') as f:
        for key, value in total_results.items():
            f.write(f"{key} : {value}\n")

    # psnr
    with open(os.path.join(output_path, output_folder_name+ "_ psnr_results.txt"), 'w') as f:
        for key, value in psnr_results.items():
            f.write(f"{key} : {value}\n")
            
    # ssim
    with open(os.path.join(output_path, output_folder_name+ "_ ssim_results.txt"), 'w') as f:
        for key, value in ssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # msssim
    with open(os.path.join(output_path, output_folder_name+ "_ msssim_results.txt"), 'w') as f:
        for key, value in msssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_vgg
    with open(os.path.join(output_path, output_folder_name+ "_ lpips_vgg_results.txt"), 'w') as f:
        for key, value in lpips_vgg_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_alex
    with open(os.path.join(output_path, output_folder_name+ "_ lpips_alex_results.txt"), 'w') as f:
        for key, value in lpips_alex_results.items():
            f.write(f"{key} : {value}\n")


def collect_memory(folder_list, output_path):
    
    total_memory = {}
    
    for folder in folder_list:
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(model_path):
            continue

        #model path에 있는 폴더 리스트
        model_folder_list = os.listdir(model_path)
        
        # 이름순으로 정렬
        model_folder_list.sort()
        
        # 가장 마지막 폴더
        model_folder = model_folder_list[-1]
        
        # 총 경로
        total_path = os.path.join(model_path, model_folder)
        
        # 해당 폴더가 포함하는 파일의 용량 총 합을 MB 단위로 출력
        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)
        print(f"{folder} : {total_size} MB")
        
        total_memory[folder] = total_size
    
    output_folder_name = output_path.split("/")[-1]
        
    # txt 파일로 저장
    with open(os.path.join(output_path, output_folder_name+ "_ total_memory.txt"), 'w') as f:
        for key, value in total_memory.items():
            f.write(f"{key} : {value}\n")
        
        
if __name__ == "__main__":
        
    parser = ArgumentParser(description="collection parameters")
    
    parser.add_argument('--output_path', type=str, default="./output/dynerf_anchor")
    parser.add_argument('--dataset', type=str, default="dycheck")
    

    args = parser.parse_args(sys.argv[1:])

    folder_list = get_folder_list(args.dataset)

    collect_metric(folder_list, args.output_path)

    collect_memory(folder_list, args.output_path)






