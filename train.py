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
import numpy as np
import random
import os, sys

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import prefilter_voxel, render, render_original, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, gvc_params):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # viewpoint stack이 없고 dataloader가 없다면
        # dnerf's branch
        # dynerf는 이거 안 탐
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            # dynerf는 이거 안 탐
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            # dynerf's branch
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            #viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init: # opt.zerostamp_init = False여서 안탐
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try: # 이 아래로는 타지 않음
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # GVC에서는 SH 안씀
        #if iteration % 1000 == 0:
        #    gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=16,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print("length of viewpoint_cams:",len(viewpoint_cams)) # 2
        # print("length of viewpoint_stack:",len(viewpoint_stack)) # 5529
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        # 
        ######## 여기서부터가 실제적인 loop #########
        #         
        for viewpoint_cam in viewpoint_cams: 
            # viewpoint_cam은 Camear() class의 인스턴스의 list, 이 중 하나씩 꺼냄
            # Camera() is a class that contains the camera parameters


            if gvc_params["GVC_testmode"] == 0:
                # original 4DGS code
                render_pkg = render_original(viewpoint_cam, gaussians, pipe, 
                                    background, stage=stage,cam_type=scene.dataset_type)
            else:             
                # testmode 1: initial_frame: scaffold-GS, others:

                """
                pre-filtering voxel
                """
            
                voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
                #voxel_visible_mask=None
                #print(f"voxel_visible_mask: {voxel_visible_mask.sum()}")
                retain_grad = (iteration < opt.update_until and iteration >= 0) 

                render_pkg = render(gvc_params, viewpoint_cam, gaussians, pipe, 
                                    background, stage=stage,cam_type=scene.dataset_type, 
                                    visible_mask=voxel_visible_mask, retain_grad=retain_grad)


            # gaussians: scene.guaussian_model.GaussianModel 오브젝트
            # pipe: pipeline_params.PipelineParams 오브젝트
            # background: torch.tensor([0,0,0]) or torch.tensor([1,1,1])
            # stage: "coarse" or "fine"
            # cam_type: "PanopticSports" or "Matterport3D"?? --> "dynerf"임
            
            # render_pkg: dict object that contains the rendered image, viewspace_point_tensor, visibility_filter, radii
            # rendered_image: torch.tensor([3, 256, 256])
            # viewspace_point_tensor: torch.tensor([3, 256, 256])
            # visibility_filter: torch.tensor([256, 256])
            # radii: torch.tensor([256, 256])
            # 얘를 아래와 같이 옮겨 담음
            
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity \
                = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                    render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            '''
            shape of image: torch.Size([3, 1014, 1352]) [o]
            shape of viewspace_point_teㅣnsor: torch.Size([37353, 3]) [o]
            shape of visibility_filter: torch.Size([227160]) [x] [37353] 나와야 됨
            shape of radii: torch.Size([227160]) [x] [37353] 나와야 됨
            '''
            images.append(image.unsqueeze(0))

            if scene.dataset_type!="PanopticSports": # 단순히 dataset 구조에 따른 차이
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        '''
        print("size of radii_list:",len(radii_list)) # 2
        print("shape of radii_list[0]:",radii_list[0].shape) # torch.size([1, 227160])
        print("shape of radii_list[1]:",radii_list[1].shape) # torch.size([1, 222524])
        print("shape of visibility_filter_list[0]:",visibility_filter_list[0].shape) # torch.size([1, 227160])
        print("shape of visibility_filter_list[1]:",visibility_filter_list[1].shape) # torch.size([1, 222524])
        print("shape of viewspace_point_tensor_list[0]:",viewspace_point_tensor_list[0].shape) # torch.size([37353, 3])
        print("shape of viewspace_point_tensor_list[1]:",viewspace_point_tensor_list[1].shape) # torch.size([37353, 3])
        '''

        # radii_list 중 가장 짧은 길이를 가진 tensor의 길이를 구함
        # min_len = min([radii.shape[1] for radii in radii_list])

        # print(f'min_len: {min_len}')

        # min_len만큼의 길이를 가진 tensor로 자름 
        # radii_list = [radii[:, :min_len] for radii in radii_list]
        # visibility_filter_list = [visibility_filter[:, :min_len] for visibility_filter in visibility_filter_list]
        # viewspace_point_tensor_list = [viewspace_point_tensor[:min_len] for viewspace_point_tensor in viewspace_point_tensor_list]

        #print("viewspace_point_tensor_list[0] grad", viewspace_point_tensor_list[0].grad) # 이때까진 grad가 None이 맞음
        #print("viewspace_point_tensor_list[1] grad", viewspace_point_tensor_list[1].grad)

        # radii = torch.cat(radii_list,0).max(dim=0).values # 뒤로 보냄
        # visibility_filter = torch.cat(visibility_filter_list).any(dim=0) # 뒤로 보냄
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        # coarse stage에서는 loss를 L1 하나만 사용함
        # fine stage이면서 time_smoothness_weight가 0이 아니면 tv_loss를 추가함
        # lambda가 주어지면 ssim_loss를 추가함
        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
            
        if stage == "fine" and gvc_params["GVC_Dynamics"] != 0:
            if opt.dynamics_loss == "entropy":
                dynamics_loss = torch.mean(-torch.sigmoid(gaussians._dynamics)*torch.log(torch.sigmoid(gaussians._dynamics)))
                loss += opt.lambda_dynamics * dynamics_loss
            elif opt.dynamics_loss == "mean":
                dynamic_mask_loss = torch.mean((torch.sigmoid(gaussians._dynamics)))
                loss += opt.lambda_dynamics * dynamic_mask_loss # Compact 3DGS 참조
            
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        # Function _RasterizeGaussiansBackward returned an invalid gradient at index 1 - got [222524, 3] but expected shape compatible with [37353, 3]
        # --> screen space point를 xyz로 해야하는데 get_xyz로 해서 생긴 문제


        min_len = min([radii.shape[1] for radii in radii_list])

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        #viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        # [min_len, 3] 의 zero torch tensor 생성
        viewspace_point_tensor_grad = torch.zeros([min_len, 3], device="cuda")

        for idx in range(0, len(viewspace_point_tensor_list)):
            #print(viewspace_point_tensor_grad, viewspace_point_tensor_list[idx].grad)
            #viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad[:min_len]
            # The size of tensor a (222524) must match the size of tensor b (227160) at non-singleton dimension 0
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save 
            timer.pause()
            if gvc_params["GVC_testmode"] == 0:
                training_report(tb_writer, iteration, Ll1, loss, 
                l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, 
                scene, render, [pipe, background], stage, scene.dataset_type)
            elif gvc_params["GVC_testmode"] >= 1:
                training_report(tb_writer, iteration, Ll1, loss, 
                l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, 
                scene, render, [pipe, background], stage, scene.dataset_type, gvc_params)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, 
                                                stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type, gvc_params)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, 
                                                stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type, gvc_params)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            
            
            # Densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            
            '''
            if iteration < opt.densify_until_iter :

                # min_len만큼의 길이를 가진 tensor로 자름 
                radii_list = [radii[:, :min_len] for radii in radii_list]
                visibility_filter_list = [visibility_filter[:, :min_len] for visibility_filter in visibility_filter_list]
                radii = torch.cat(radii_list,0).max(dim=0).values
                visibility_filter = torch.cat(visibility_filter_list).any(dim=0) 

                # Keep track of max radii in image-space for pruning
                print("max radii2D shape", gaussians.max_radii2D.shape) # torch.Size([37353])
                print("radii shape", radii.shape) # torch.Size([222524])
                print("visibility_filter shape", visibility_filter.shape) # torch.Size([222524])

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
            '''
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, gvc_params):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)

    # Create Gaussian model
    
    if gvc_params["GVC_testmode"] == 0:
    # original 4DGS code
        gaussians = GaussianModel(dataset.sh_degree, hyper)
    elif gvc_params["GVC_testmode"] == 1:
    # testmode 1: initial_frame: scaffold-GS, others: 4DGS
        gaussians = GaussianModel(hyper, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, 
                                  dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, 
                                  dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, 
                                  dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, gvc_params)
        # print(dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist) # False, False, False
    elif gvc_params["GVC_testmode"] == 2 or gvc_params["GVC_testmode"] == 3: 
    # testmode 2: initial_frame: scaffold-GS, deformation: anchor points and local context features
        gaussians = GaussianModel(hyper, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, 
                                  dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, 
                                  dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, 
                                  dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, gvc_params)
        # print(dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist) # False, False, False
    else:
        raise ValueError("Unsupported GVC testmode")

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    # scene is a class that contains attributes and method: gaussians, dataset, save, load, getTrainCameras, getTestCameras, getVideo
    timer.start()
    # 기본적인 instance 생성 및 변수 할당 후, scene_reconstruction 함수를 coarse, fine stage에 대해 각각 1번씩 실행
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer, gvc_params)

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, gvc_params)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, gvc_params):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    if gvc_params["GVC_testmode"] == 0:
                        image = torch.clamp(renderFunc(viewpoint, 
                        scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    elif gvc_params["GVC_testmode"] >= 1:
                        image = torch.clamp(renderFunc(gvc_params, viewpoint, 
                        scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    else:
                        raise ValueError("Non-supported GVC testmode")

                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")

    # my test args
    parser.add_argument("--GVC_testmode", type=int, default = 1)
    parser.add_argument("--GVC_Scale_Activation", type=int, default = 1, help="0: default, 1: scale activation outside")
    parser.add_argument("--GVC_Opacity_Activation", type=int, default = 0, help="0: default, 1: opacity activation outside")
    parser.add_argument("--GVC_Dynamics", type=int, default = 1, help="0: None, 1: dynamics(all), 2: dynamic: anchor only, 3: dynamic: local context only, 4: dynamic: offset only, 5: anchor and feature, 6: anchor and offset")
    parser.add_argument("--GVC_Dynamics_type", type=str, default = "mask", help="mul, mask")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)
    #print("s.kwak check:" + str(args.net_width)) # checked

    gvc_params = lp.extract(args).gvc_params
    
    print("GVC_testmode: ??", gvc_params["GVC_testmode"])
    
    # append GVC testmode to gvc_params
    
    gvc_params["GVC_testmode"] = args.GVC_testmode
    gvc_params["GVC_Scale_Activation"] = args.GVC_Scale_Activation
    gvc_params["GVC_Opacity_Activation"] = args.GVC_Opacity_Activation
    gvc_params["GVC_Dynamics"] = args.GVC_Dynamics
    gvc_params["GVC_Dynamics_type"] = args.GVC_Dynamics_type

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 기본적으로 training 함수를 실행함
    # dataset, hyper, opt, pipe,
    #training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)


    print("---------------------------------")
    print("GVC_testmode: ", args.GVC_testmode)
    print("GVC_Scale_Activation: ", args.GVC_Scale_Activation)
    print("GVC_Opacity_Activation: ", args.GVC_Opacity_Activation)
    print("GVC_Dynamics: ", args.GVC_Dynamics)
    print("GVC_Dynamics_type: ", args.GVC_Dynamics_type)
    print("---------------------------------")

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, gvc_params)

    # All done
    print("\nTraining complete.")
