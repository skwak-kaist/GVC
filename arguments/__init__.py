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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        
        ############################
        # Scaffold-GS related params
        ############################
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.use_feat_bank = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        ############################
        # 4DGS-related params
        ############################
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8

        ############################
        # GVC-related params
        ############################
        self.testmode = 1 # GVC test mode version
        # testmode 1: initial_frame: scaffold-GS, others: 4DGS
        # testmode 2: initial_frame: scaffold-GS, feature deformation
        # testmode 3: dynamic mask 
        # testmode 4: temporal scaffolding
        
        self.scale_activation = 1 # default 1
        self.opacity_activation = 0 # default 0

        # dynamics mode
        self.dynamics = 1
        # 0: None, 
        # 1: dynamics(all), 
        # 2: dynamic: anchor only, 
        # 3: dynamic: local context only, 
        # 4: dynamic: offset only, 
        # 5: anchor and feature, 
        # 6: anchor and offset"
        
        self.dynamics_type = "mask"
        # mul: multiply the dynamic value to the feature
        # mask: mask the feature with the dynamic mask

        # GVC 4.0
        self.num_of_segments = 4
        # number of segments for temporal scaffolding
        
        self.temporal_scaffolding = 1 # 0: off 1: uniform scaffolding 2: adaptive scaffolding
        self.local_deform_method = "explicit" # explicit(gaussian) or implicit(feature)
        
        # GVC 5.0
        self.temporal_adjustment = 1
        self.temporal_adjustment_step_size = 0.1
        self.temporal_adjustment_threshold = 1.0

        super().__init__(parser, "Loading Parameters", sentinel)

    def merge_gvc_params(self, args):
        
        gvc_params = {}
        gvc_params["GVC_testmode"] = self.extract(args).testmode
        gvc_params["GVC_Scale_Activation"] = args.scale_activation
        gvc_params["GVC_Opacity_Activation"] = args.opacity_activation
        gvc_params["GVC_Dynamics"] = args.dynamics
        gvc_params["GVC_Dynamics_type"] = args.dynamics_type
        gvc_params["GVC_num_of_segments"] = args.num_of_segments
        gvc_params["GVC_temporal_scaffolding"] = args.temporal_scaffolding
        gvc_params["GVC_local_deform_method"] = args.local_deform_method
        gvc_params["GVC_temporal_adjustment"] = args.temporal_adjustment
        gvc_params["GVC_temporal_adjustment_step_size"] = args.temporal_adjustment_step_size
        gvc_params["GVC_temporal_adjustment_threshold"] = args.temporal_adjustment_threshold
        
        print("GVC_testmode: ", gvc_params["GVC_testmode"])
        print("GVC_Scale_Activation: ", gvc_params["GVC_Scale_Activation"])
        print("GVC_Opacity_Activation: ", gvc_params["GVC_Opacity_Activation"])
        print("GVC_Dynamics: ", gvc_params["GVC_Dynamics"])
        print("GVC_Dynamics_type: ", gvc_params["GVC_Dynamics_type"])
        print("GVC_num_of_segments: ", gvc_params["GVC_num_of_segments"])
        print("GVC_temporal_scaffolding: ", gvc_params["GVC_temporal_scaffolding"])
        print("GVC_local_deform_method: ", gvc_params["GVC_local_deform_method"])
        print("GVC_temporal_adjustment: ", gvc_params["GVC_temporal_adjustment"])
                        
        return gvc_params

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
        
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        
        self.kplanes_config_local = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires_local = [1, 2, 4, 8] # multi resolution of voxel grid
        
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        # for scaffold-GS
        self.anchor_deform = True
        self.local_context_feature_deform = True
        self.grid_offsets_deform = True
        self.grid_scale_deform = True
        
        self.deform_feat_dim = 32
        self.deform_n_offsets = 10
        
        # dynamics
        self.dynamics_activation='sigmoid' # relu or leakyrelu or sigmoid
        
        # dynamics loss weight
        self.dynamics_loss_weight = 0.01
        

        
        super().__init__(parser, "ModelHiddenParams")

        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        ############################
        # 4DGS-related params
        ############################
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size=1
        self.add_point=False

         
        ############################
        # Scaffold-GS-related params
        ############################
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        # dynamic masks
        self.dynamics_lr = 0.01
        self.dynamics_lr_init = 0.01
        self.lambda_dynamics = 0.0002
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        # for temporal adjustment
        self.temporal_adjustment_until = 45_000
        self.temporal_adjustment_from = 3000
        self.temporal_adjustment_interval = 1000
            
        # for dynamics
        #self.dynamics_loss = "entropy"
        self.dynamics_loss = None

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
