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

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # visible_mask에 True가 몇 개인지 확인
    # print("visible_mask:",visible_mask.sum()) # 너무 큼

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    #print("anchor device:",anchor.device) # cuda:0
    #print("viewpoint_camera.camera_center device:",viewpoint_camera.camera_center.device) # cpu
    viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        #camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10 # 왜 인지는 모르겠지만 이렇게 하니까 cuda error가 발생하지 않음
            
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        # print("cat_local_view_wodist:",cat_local_view_wodist.shape) # 얘는 죄가 없음
        #print("camera_indicies:",camera_indicies) # camera_indicies: torch.Size([10562])
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist) 
        # 여기서 계속 CUDA error 발생함
        # CUBLAS_STATUS_EXECUTION_FAILED --> dimension mistmatch이슈가 많다고 함
        # cat_local_view_wodist: shape torch.size([10562, 35])


    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    #print("mask:",mask.shape) # mask: torch.Size([105620])
    #print("neural_opacity:",neural_opacity.shape) # neural_opacity: torch.Size([105620, 1])

    # select opacity 
    opacity = neural_opacity[mask]
    # neural_opacity: 105620, 1
    # mask: 105620

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    # modified!!! - s.kwak
    #rot = pc.rotation_activation(scale_rot[:,3:7])
    rot = scale_rot[:,3:7]
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def render(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):

    # common pass of render function
    # GVC testmode에 따라서 render 함수를 호출하는 방식이 달라짐

    is_training = pc.get_color_mlp.training

    if gvc_params["GVC_testmode"] == 0:
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_original(
                viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_original(
                viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
            
    elif gvc_params["GVC_testmode"] == 1:
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test1(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test1(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)


    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "depth":depth,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                }

# GVC test mode 1일 때 호출되는 render 함수
def render_test1(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
              override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    is_training = pc.get_color_mlp.training
           
    # anchor 인근의 neural gaussian을 생성한다. 
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training) 

    # screenspace point 생성
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # mean3D 부터 먼저 할당
    means3D = xyz

    # Set up rasterization configuration
    if cam_type != "PanopticSports": # 
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            #sh_degree=pc.active_sh_degree,
            sh_degree=1,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera'] # PanopticSports인 경우에는 camera에 다 들어있나보다
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Scaffold-GS
    means2D = screenspace_points 
    # opacity = opacity # neural gaussian에서 생성된 opacity
    shs = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        '''
        # original 4DGS
        scales = pc._scaling
        rotations = pc._rotation
        '''
        # Scaffold-GS
        scales = scaling
        rotations = rot

    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        # print("coarse stage: no deformation")
        # print("means3D_final, scales_final, rotations_final, opacity_final, shs_final",means3D_final.shape, scales_final.shape, rotations_final.shape, opacity_final.shape, shs_final.shape) 
        # torch.Size([37353, 3]) torch.Size([37353, 3]) torch.Size([37353, 4]) torch.Size([37353, 1]) torch.Size([37353, 16, 3])
        # Keyframe의 Gaussian attributes torch tensor들을 가지고 있다고 보면 됨
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
        # time2 = get_time()
        # print("asset value:",time2-time1)
        # 각각의 attributes에 대한 사전 정의된 activation function을 적용
        # Scaffold-GS에는 activation function이 없으므로 fine stage에서만 적용

    else:
        raise NotImplementedError

    # fine elif 문 안에 있던걸 밖으로 뺐음

    #scales_final = pc.scaling_activation(scales_final) 
    rotations_final = pc.rotation_activation(rotations_final)
    #opacity_final = pc.opacity_activation(opacity_final)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = color,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    if is_training:
        return rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth
    else:
        return rendered_image, screenspace_points, radii, depth
    


# original render function
def render_original(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports": # 
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera'] # PanopticSports인 경우에는 camera에 다 들어있나보다
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points 
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        # print("coarse stage: no deformation")
        # print("means3D_final, scales_final, rotations_final, opacity_final, shs_final",means3D_final.shape, scales_final.shape, rotations_final.shape, opacity_final.shape, shs_final.shape) 
        # torch.Size([37353, 3]) torch.Size([37353, 3]) torch.Size([37353, 4]) torch.Size([37353, 1]) torch.Size([37353, 16, 3])
        # Keyframe의 Gaussian attributes torch tensor들을 가지고 있다고 보면 됨
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    # 각각의 attributes에 대한 사전 정의된 activation function을 적용
    scales_final = pc.scaling_activation(scales_final) 
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):

    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    #print("viewpoint_camera.image_height_device:",viewpoint_camera.image_height.device) # cpu
    #print("viewpoint_camera.image_width_device:",viewpoint_camera.image_width.device) # cpu
    #print("viewpoint_camera.world_view_transform_device:",viewpoint_camera.world_view_transform.device) # cpu
    #print("viewpoint_camera.full_proj_transform_device:",viewpoint_camera.full_proj_transform.device) # cpu
    #print("viewpoint_camera.camera_center_device:",viewpoint_camera.camera_center.device) # cpu

    # cuda로 옮겨줌
    viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.cuda()
    viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.cuda()
    viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()  

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # print(f'Mean 3D 1~30 (before): {means3D[:30]}')

    # temporal varibale copy
    # visible filter의 이상한 동작을 막기 위해 카피함 >> 해도 마찬가지임
    # means3D_temp = means3D.clone()
    # scales_temp = scales.clone()
    # rotations_temp = rotations.clone()
    # cov3D_precomp_temp = cov3D_precomp.clone()

    # means3D가 메모리에 있는지 확인
    #print("means3D device:",means3D.device) # cuda:0

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    #print(f'Mean 3D 1~30 (after): {means3D[:30]}')
    #print(f'Radii Pure 1~30: {radii_pure[:30]}')

    return radii_pure > 0
    
