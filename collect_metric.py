
import os, sys
import json
from argparse import ArgumentParser, Namespace

def get_folder_list(dataset):
    if dataset == "dycheck":
        # apple, backpack, block, creeper, handwavy, haru-sit, mochi-high-five, pillow, space-out, spin, sriracha-tree, teddy, wheel
        # folder_list = ["apple", "backpack", "block", "creeper", "handwavy", "haru-sit", "mochi-high-five", "pillow", "space-out", "spin", "sriracha-tree", "teddy", "wheel"]
        # apple, block, paper-windmill, space-out, spin, teddy, wheel
        folder_list = ["apple", "block", "spin", "paper-windmill", "space-out", "teddy", "wheel"]
        
    elif dataset == "dynerf":
        # coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
        folder_list = ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"]
    elif dataset == "nvidia":
        # coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
        folder_list = ["Balloon1", "Balloon2", "Jumping", "dynamicFace","Playground", "Skating", "Truck", "Umbrella"]
    elif dataset == "etri4dgs":
        # coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
        folder_list = ["Bartender", "ClappingGame"]
    elif dataset == "hypernerf":
        # interp_aleks-teapot chickchicken cut-lemon1 hand1 slice-banana torchocolate
        # misc_americano cross-hands1 espresso keyboard oven-mitts split-cookie tamping
        # vrig_3dprinter broom chicken peel-banana
        folder_list = ["interp_aleks-teapot", "interp_chickchicken", "interp_cut-lemon1", "interp_hand1", "interp_slice-banana", "interp_torchocolate", 
                       "misc_americano", "misc_cross-hands1", "misc_espresso", "misc_keyboard", "misc_oven-mitts", "misc_split-cookie", "misc_tamping", 
                       "vrig_3dprinter", "vrig_broom", "vrig_chicken", "vrig_peel-banana"]
        
        #folder_list = ["aleks-teapot", "chickchicken", "cut-lemon1", "hand1", "slice-banana", "torchocolate", 
        #               "americano", "cross-hands1", "espresso", "keyboard", "oven-mitts", "split-cookie", "tamping", 
        #               "3dprinter", "broom", "chicken", "peel-banana"]
    elif dataset == "dnerf":
        # bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex
        folder_list = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]
    elif dataset == "panoptic_sports":
        # basketball boxes football juggle softball tennis
        folder_list = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]

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
        # print(f"{folder} : {results[result_key]['PSNR']}")
    
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
    with open(os.path.join(output_path, output_folder_name+ "_psnr_results.txt"), 'w') as f:
        for key, value in psnr_results.items():
            f.write(f"{key} : {value}\n")
            
    # ssim
    with open(os.path.join(output_path, output_folder_name+ "_ssim_results.txt"), 'w') as f:
        for key, value in ssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # msssim
    with open(os.path.join(output_path, output_folder_name+ "_msssim_results.txt"), 'w') as f:
        for key, value in msssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_vgg
    with open(os.path.join(output_path, output_folder_name+ "_lpips_vgg_results.txt"), 'w') as f:
        for key, value in lpips_vgg_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_alex
    with open(os.path.join(output_path, output_folder_name+ "_lpips_alex_results.txt"), 'w') as f:
        for key, value in lpips_alex_results.items():
            f.write(f"{key} : {value}\n")
   
def collect_psnr_ssim_lpips_memory(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]
    
    # 결과 파일 생성
    with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]      
        
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['SSIM']
        lpips_results[folder] = results[result_key]['LPIPS-vgg']

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
        # print(f"{folder} : {total_size} MB")
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB")

        # txt 파일로 저장
        with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB\n")
            
    

def collect_psnr_msssim_lpips_memory(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]
    
    # 결과 파일 생성
    with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]      
        
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['MS-SSIM']
        lpips_results[folder] = results[result_key]['LPIPS-vgg']

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
        # print(f"{folder} : {total_size} MB")
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['MS-SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB")

        # txt 파일로 저장
        with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['MS-SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB\n")



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
        # print(f"{folder} : {total_size} MB")
        
        # output_path/folder에 deform이라는 폴더가 있는지 검사
        # deform 폴더가 있는 경우 해당 폴더의 용량을 더함
        deform_path = os.path.join(output_path, folder, "deform")
        if os.path.exists(deform_path):
            deform_folder_list = os.listdir(deform_path)
            deform_folder_list.sort()
            deform_folder = deform_folder_list[-1]
            deform_total_path = os.path.join(deform_path, deform_folder)
            deform_total_size = sum(os.path.getsize(os.path.join(deform_total_path, f)) for f in os.listdir(deform_total_path)) / (1000*1000)
            total_size += deform_total_size
        
        
        total_memory[folder] = total_size
    
    output_folder_name = output_path.split("/")[-1]
        
    # txt 파일로 저장
    with open(os.path.join(output_path, output_folder_name+ "_ total_memory.txt"), 'w') as f:
        for key, value in total_memory.items():
            f.write(f"{key} : {value}\n")

def merge_psnr_and_memory(folder_list, output_path):
    
    psnr_results = {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]
    
    # 결과 파일 생성
    with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]      
        

        psnr_results[folder] = results[result_key]['PSNR']

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
        # print(f"{folder} : {total_size} MB")
        
        # output_path/folder에 deform이라는 폴더가 있는지 검사
        # deform 폴더가 있는 경우 해당 폴더의 용량을 더함
        deform_path = os.path.join(output_path, folder, "deform")
        if os.path.exists(deform_path):
            deform_folder_list = os.listdir(deform_path)
            deform_folder_list.sort()
            deform_folder = deform_folder_list[-1]
            deform_total_path = os.path.join(deform_path, deform_folder)
            deform_total_size = sum(os.path.getsize(os.path.join(deform_total_path, f)) for f in os.listdir(deform_total_path)) / (1000*1000)
            total_size += deform_total_size
        
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {total_size} MB")

        # txt 파일로 저장 (계속 이어 쓰기)
        with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {total_size} MB\n")
        

def merge_masked_results(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]
    
    # 결과 파일 생성
    with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'w') as f:
        f.write("")
    
    with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'w') as f:
        f.write("")
        
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results_masked.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")               
            continue
            
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]      

        psnr_results[folder] = results[result_key]['mPSNR']
        ssim_results[folder] = results[result_key]['mSSIM']
        lpips_results[folder] = results[result_key]['mLPIPS']

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
        # print(f"{folder} : {total_size} MB")
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['mPSNR']} {total_size} MB")

        # txt 파일로 저장 (계속 이어 쓰기)
        with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mPSNR']} {total_size} MB\n")    
        
        with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mPSNR']} {results[result_key]['mSSIM']} {results[result_key]['mLPIPS']} {total_size} MB\n")    


def merge_masked_lpips_vgg(folder_list, output_path):
    
    lpips_results= {}
    
    output_folder_name = output_path.split("/")[-1]
    
    # 결과 파일 생성
    with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results_masked_lpips.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        # 해당 폴더가 없는 경우 pass
        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
        
        # read the json
        with open(json_path) as f:
            results = json.load(f)

        # results의 최 상단 key값이 무엇인지 확인
        result_key = list(results.keys())[0]      

        lpips_results[folder] = results[result_key]['mLPIPS']
       
        print(f"{folder} : {results[result_key]['mLPIPS']}")

        # txt 파일로 저장 (계속 이어 쓰기)
        with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mLPIPS']}\n")
        
   
        
        
if __name__ == "__main__":
        
    parser = ArgumentParser(description="collection parameters")
    
    parser.add_argument('--output_path', type=str, default="./output/dynerf_anchor")
    parser.add_argument('--dataset', type=str, default="dycheck")
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--lpips_only', type=int, default=0)
    parser.add_argument('--all', type=int, default=0)
    

    args = parser.parse_args(sys.argv[1:])

    folder_list = get_folder_list(args.dataset)

    #collect_metric(folder_list, args.output_path)

    #collect_memory(folder_list, args.output_path)
    
    collect_psnr_ssim_lpips_memory(folder_list, args.output_path)
    
    collect_psnr_msssim_lpips_memory(folder_list, args.output_path)


    if args.mask:
	    if args.lpips_only:
	        merge_masked_lpips_vgg(folder_list, args.output_path)
	    else:        
	        merge_masked_results(folder_list, args.output_path)
    else:
	    #merge_psnr_and_memory(folder_list, args.output_path)
        collect_psnr_ssim_lpips_memory(folder_list, args.output_path)
 		
    
    




