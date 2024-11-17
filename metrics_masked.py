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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim

from dycheck.core import metrics
from torch2jax import j2t, t2j 
import numpy as np

def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        
        image_names.append(fname)
    
    for fname in os.listdir(mask_dir):
        mask = Image.open(os.path.join(mask_dir , fname))
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :].cuda())
        
    return renders, gts, masks, image_names

def evaluate(model_paths, data_paths, lpips_only):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    #lpips_only = True
    
    compute_lpips = metrics.get_compute_lpips("vgg") 

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            mask_dir = os.path.join(data_paths[0], "covisible", "2x", "val")
            
            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                                            
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, mask_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                
                mssims = []
                mpsnrs = []
                mlpipss = []
                
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                #for idx in tqdm(range(10), desc="Metric evaluation progress"):
                    
                    # pre-process
                    render = t2j(torch.squeeze(renders[idx], 0).transpose(0,2))
                    gt = t2j(torch.squeeze(gts[idx], 0).transpose(0,2))
                    mask = t2j(torch.squeeze(masks[idx], 0).transpose(0,2))
                    
                    # calculate metrics
                    if lpips_only:
                        mlpips = compute_lpips(render, gt, mask)
                    else:
                        mssim = metrics.compute_ssim(render, gt, mask) 
                        mpsnr = metrics.compute_psnr(render, gt, mask)
                        mlpips = compute_lpips(render, gt, mask)
                        
                    # skip old metrics
                    '''
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1-ms_ssims[-1])/2)
                    '''
                    if lpips_only:
                        mlpipss.append(mlpips)    
                    else:
                        mssims.append(mssim)
                        mpsnrs.append(mpsnr)
                        mlpipss.append(mlpips)
                    
                '''
                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                            }
                                                        )
                '''
                
                if lpips_only:
                    print("Scene: ", scene_dir,  "mLPIPS: {:>12.7f}".format(torch.tensor(np.array(mlpipss)).mean(), ".5"))
                    
                    full_dict[scene_dir][method].update({"mLPIPS": torch.tensor(np.array(mlpipss)).mean().item(),},

                                                        )
                    per_view_dict[scene_dir][method].update({"mLPIPS": {name: mlp for mlp, name in zip(torch.tensor(np.array(mlpipss)).tolist(), image_names)},}
                                                            )
                
                
                else:
                                    
                    print("Scene: ", scene_dir,  "mSSIM : {:>12.7f}".format(torch.tensor(np.array(mssims)).mean(), ".5"))
                    print("Scene: ", scene_dir,  "mPSNR : {:>12.7f}".format(torch.tensor(np.array(mpsnrs)).mean(), ".5"))
                    print("Scene: ", scene_dir,  "mLPIPS-vgg: {:>12.7f}".format(torch.tensor(np.array(mlpipss)).mean(), ".5"))
                    
                    full_dict[scene_dir][method].update({"mSSIM": torch.tensor(np.array(mssims)).mean().item(),
                                                            "mPSNR": torch.tensor(np.array(mpsnrs)).mean().item(),
                                                            "mLPIPS": torch.tensor(np.array(mlpipss)).mean().item(),},

                                                        )
                    per_view_dict[scene_dir][method].update({"mSSIM": {name: mssim for mssim, name in zip(torch.tensor(np.array(mssims)).tolist(), image_names)},
                                                                "mPSNR": {name: mpsnr for mpsnr, name in zip(torch.tensor(np.array(mpsnrs)).tolist(), image_names)},
                                                                "mLPIPS": {name: mlp for mlp, name in zip(torch.tensor(np.array(mlpipss)).tolist(), image_names)},}
                                                            )
            if lpips_only:
                with open(scene_dir + "/results_masked_lpips.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view_masked_lpips.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)                
                
            else: 
                with open(scene_dir + "/results_masked.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view_masked.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
                    
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--data_paths' ,'-d', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--lpips_only', type=int, default=0)
    args = parser.parse_args()
    evaluate(args.model_paths, args.data_paths, args.lpips_only)
    
