# dycheck 기준 v5.0.8.0.0.t1-2임

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     #'resolution': [64, 64, 64, 150]
     'resolution': [32, 32, 32, 10]
    },
#    multires = [1,2,4],
    multires = [1,2],
    
    kplanes_config_local = { # gaussian attributes deformation k-planes
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 75]
     #'resolution': [64, 64, 64, 100]
    },
#    multires = [1,2,4],
    multires_local = [1,2],
    
    defor_depth = 0,
    net_width = 64,
    plane_tv_weight = 0.0001,
    time_smoothness_weight = 0.01,
    l1_time_planes =  0.0001,
    render_process=False,
    weight_decay_iteration=0,
    bounds=1.6,

	# 모두 deform     
    anchor_deform = True,
    local_context_feature_deform = True,
    grid_offsets_deform = True,
    grid_scale_deform = True,
    
    deform_feat_dim = 32, # ModelParams의 feat_dim과 일치시켜야 함
    deform_n_offsets = 10, # ModelParams의 n_offsets과 일치시켜야 함    
    dynamics_activation = "sigmoid", 
    

)

ModelParams = dict(
    appearance_dim = 16,
    n_offsets = 10, 
    use_feat_bank = True, 
    #voxel_size = 0.0005,
    voxel_size = 0.01, # voxel size를 늘림 0.001 --> 0.01
    feat_dim = 32, # ModelHiddenParams의 feat_dim과 일치시켜야 함
    update_init_factor = 4,
    
    scale_activation = 1,
	opacity_activation = 0,
    
    testmode = 5,
	dynamics = 1, 
	# 0: None, 1: dynamics(all), 2: dynamic: anchor only, 3: dynamic: local context only, 
	# 4: dynamic: offset only, 5: anchor and feature, 6: anchor and offset
	dynamics_type = "mask_mul", # mask or mul or mask_mul or mul_mask
    
    temporal_scaffolding = 1,
    num_of_segments = 2, 
    local_deform_method = "explicit",
 
 	# temporal adjustment
    temporal_adjustment = 1,
    temporal_adjustment_step_size = 0.1,
    temporal_adjustment_threshold = 1.5,
 
)

OptimizationParams = dict(
    # dataloader=True,
    iterations = 20000, 
    batch_size=2,
    coarse_iterations = 5000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    pruning_interval = 8000,
    percent_dense = 0.01,
    densify_until_iter = 10_000,
    opacity_reset_interval = 300000,
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
    
    # Scaffold-GS related params
    start_stat = 500,
    update_from = 1500,
    update_interval = 100,
    update_until = 7000,

# my test parameters
    min_opacity = 0.001,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0002,

	#dynamics_loss = "mean", # mean or entropy or mean_entropy or entropy_mean, 그 외의 값을 주면 Loss를 걸지 않음	
	dynamics_loss = "none",

	# temporal adjustment
    temporal_adjustment_until = 10000,
    temporal_adjustment_from = 5000,
    temporal_adjustment_interval = 1000,


)

