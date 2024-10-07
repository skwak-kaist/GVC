ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,

)

ModelParams = dict(
    appearance_dim = 32,
    n_offsets = 10, 
    use_feat_bank = False, 
    voxel_size = 0.001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 14000,
    batch_size=2, # 원래 4였는데 2로 다 통일시킴
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    min_opacity = 0.005,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0002,
    
    
    # pruning_interval = 2000
)
