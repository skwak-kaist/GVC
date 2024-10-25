# 기준 3번 config: v3.1.0.1.0.

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     #'resolution': [64, 64, 64, 150]
#     'resolution': [32, 32, 32, 10]
	'resolution': [16, 16, 16, 10],
    },
#    multires = [1,2,4],
    multires = [1,2],
    
    kplanes_config_local = { # gaussian attributes deformation k-planes
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
     #'resolution': [64, 64, 64, 100]
    },
#    multires = [1,2,4],
    multires_local = [1,2],
    
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=True,

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
    use_feat_bank = False, 
    #voxel_size = 0.0005,
    voxel_size = 0.001, # voxel size를 줄여봄. 기본값은 0.001
    feat_dim = 32, # ModelHiddenParams의 feat_dim과 일치시켜야 함
    
    scale_activation = 1,
	opacity_activation = 0,
    
    testmode = 4,
	dynamics = 1, 
	# 0: None, 1: dynamics(all), 2: dynamic: anchor only, 3: dynamic: local context only, 
	# 4: dynamic: offset only, 5: anchor and feature, 6: anchor and offset
	dynamics_type = "mul", # mask or mul or mask_mul or mul_mask
    
    temporal_scaffolding = 1,
    num_of_segments = 16, 
    local_deform_method = "explicit",
 
)

OptimizationParams = dict(
    # dataloader=True,
    iterations = 60_000,
    batch_size=2,
    coarse_iterations = 3000,
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
    update_until = 15_000,

# my test parameters
    min_opacity = 0.005,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0002,

	#dynamics_loss = "mean", # mean or entropy or mean_entropy or entropy_mean, 그 외의 값을 주면 Loss를 걸지 않음	
	dynamics_loss = "none",


)

