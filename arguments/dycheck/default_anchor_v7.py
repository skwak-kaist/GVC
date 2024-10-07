ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4],
#    multires = [1,2],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=True,
)

ModelParams = dict(
    appearance_dim = 16,
    n_offsets = 10, 
    use_feat_bank = True, 
    #voxel_size = 0.0005,
    voxel_size = 0.0001, # voxel size를 줄여봄. 기본값은 0.001
    
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
    #update_until = 15_000
    update_until = 60_000,
        
#    min_opacity = 0.005,
    min_opacity = 0.001,
#    success_threshold = 0.8,
    success_threshold = 0.9,
    densify_grad_threshold = 0.0002,
)
