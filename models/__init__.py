def get_model(params, brain_regions, sorted_indices):
    if params.model == 'CSBrain':
        from .CSBrain import CSBrain
        model = CSBrain(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions,
            sorted_indices
        )
    elif params.model == 'OurModel':
        from .our_model import OurModel
        model = OurModel(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
        )
    elif params.model == 'ChannelCausal':
        from .channel_causal import ChannelCausal
        model = ChannelCausal(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions,
            sorted_indices
        )
    elif params.model == 'LLMVQ':
        from .llm_vq import CSBrainLLMVQ
        model = CSBrainLLMVQ(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions,
            sorted_indices
        )
    elif params.model == 'Align':
        from .alignment import CSBrainAlign
        _mbp = getattr(params, 'mamba_band_periods', None)
        if isinstance(_mbp, str) and _mbp:
            _mbp = eval(_mbp)
        model = CSBrainAlign(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            causal=getattr(params, 'causal', False),
            project_to_source=getattr(params, 'project_to_source', False),
            num_sources=getattr(params, 'num_sources', 32),
            source_projector_ckpt=getattr(params, 'source_projector_ckpt', None),
            freeze_source_projector=getattr(params, 'freeze_source_projector', True),
            adversarial_weight=getattr(params, 'adversarial_weight', 0.0),
            equivariance_weight=getattr(params, 'equivariance_weight', 0.0),
            info_max_weight=getattr(params, 'info_max_weight', 0.0),
            alignment_weight=getattr(params, 'alignment_weight', 1.0),
            patch_embed_type=getattr(params, 'patch_embed_type', 'cnn'),
            mamba_band_periods=_mbp,
            n_mamba_layers=getattr(params, 'n_mamba_layers', 2),
            mamba_d_state=getattr(params, 'mamba_d_state', 16),
            mamba_d_conv=getattr(params, 'mamba_d_conv', 4),
            mamba_expand=getattr(params, 'mamba_expand', 2),
        )
    elif params.model == 'WorldModel':
        from .alignment import CSBrainAlign
        from .world_model import LatentPredictor, WorldModelWrapper
        _mbp = getattr(params, 'mamba_band_periods', None)
        if isinstance(_mbp, str) and _mbp:
            _mbp = eval(_mbp)
        encoder = CSBrainAlign(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            causal=getattr(params, 'causal', False),
            project_to_source=getattr(params, 'project_to_source', False),
            num_sources=getattr(params, 'num_sources', 32),
            source_projector_ckpt=getattr(params, 'source_projector_ckpt', None),
            freeze_source_projector=getattr(params, 'freeze_source_projector', True),
            adversarial_weight=0.0,
            equivariance_weight=0.0,
            info_max_weight=0.0,
            alignment_weight=getattr(params, 'alignment_weight', 1.0),
            patch_embed_type=getattr(params, 'patch_embed_type', 'cnn'),
            mamba_band_periods=_mbp,
            n_mamba_layers=getattr(params, 'n_mamba_layers', 2),
            mamba_d_state=getattr(params, 'mamba_d_state', 16),
            mamba_d_conv=getattr(params, 'mamba_d_conv', 4),
            mamba_expand=getattr(params, 'mamba_expand', 2),
        )
        max_horizon = getattr(params, 'max_horizon', 1)
        # ``max_horizon == 0`` short-circuits the world-model components:
        # the wrapper reduces to plain CSBrainAlign (masked recon + image
        # alignment) so no predictor parameters enter the optimiser.
        if max_horizon > 0:
            predictor = LatentPredictor(
                d_model=params.d_model,
                predictor_d_model=getattr(params, 'predictor_d_model', 512),
                n_layers=getattr(params, 'predictor_n_layers', 4),
                n_heads=getattr(params, 'predictor_n_heads', 8),
                dim_feedforward=getattr(params, 'predictor_dim_feedforward', 1024),
                dropout=getattr(params, 'dropout', 0.1),
                max_horizon=max(max_horizon, 1),
            )
        else:
            predictor = None
        model = WorldModelWrapper(
            encoder=encoder,
            predictor=predictor,
            latent_pred_weight=getattr(params, 'latent_pred_weight', 1.0),
            cls_pred_weight=getattr(params, 'cls_pred_weight', 0.1),
            max_horizon=max_horizon,
            ramp_epochs=getattr(params, 'pred_ramp_epochs', 2),
        )
        return model
    elif params.model == 'Spectral':
        from .spectral_alignment import CSBrainSpectral
        model = CSBrainSpectral(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            num_bands=getattr(params, 'num_bands', 4),
            num_visual_levels=getattr(params, 'num_visual_levels', 3),
            dino_layer_indices=getattr(params, 'dino_layer_indices', (3, 7, 12)),
        )
    elif params.model == 'SourceProjector':
        from .alignment import SourceProjector
        model = SourceProjector(
            in_dim=params.in_dim,
            num_sources=getattr(params, 'num_sources', 32),
            decorr_weight=getattr(params, 'decorr_weight', 0.1),
        )
        return model
    elif params.model == 'CNN':
        from .cnn import CSBrainCNN
        model = CSBrainCNN(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            # sorted_indices
        )
    else:
        raise ValueError(f"Unknown model type: {params.model}")

    # --- Wrap with DINOv2 self-distillation if requested ---
    if getattr(params, 'dino_mode', False):
        from .dino_eeg import DINOEEGModel
        model = DINOEEGModel(
            student_backbone=model,
            d_model=getattr(params, 'd_model', params.in_dim),
            dino_head_hidden_dim=getattr(params, 'dino_head_hidden_dim', 256),
            dino_head_bottleneck_dim=getattr(params, 'dino_head_bottleneck_dim', 256),
            n_prototypes=getattr(params, 'n_prototypes', 4096),
            dino_head_n_layers=getattr(params, 'dino_head_n_layers', 3),
            student_temp=getattr(params, 'student_temp', 0.1),
            teacher_temp_base=getattr(params, 'teacher_temp_base', 0.04),
            teacher_temp_final=getattr(params, 'teacher_temp_final', 0.07),
            ema_momentum_base=getattr(params, 'ema_momentum_base', 0.992),
            ema_momentum_final=getattr(params, 'ema_momentum_final', 0.9995),
            dino_loss_weight=getattr(params, 'dino_loss_weight', 1.0),
            ibot_loss_weight=getattr(params, 'ibot_loss_weight', 1.0),
            koleo_loss_weight=getattr(params, 'koleo_loss_weight', 0.1),
            use_freq_subband=getattr(params, 'use_freq_subband', False),
            freq_n_bands=getattr(params, 'freq_n_bands', 5),
            freq_min_bands=getattr(params, 'freq_min_bands', 1),
            freq_max_bands=getattr(params, 'freq_max_bands', None),
            n_local_crops=getattr(params, 'n_local_crops', 4),
            local_crop_time_scale=eval(getattr(params, 'local_crop_time_scale', '(0.3, 0.7)')),
            local_crop_channel_scale=eval(getattr(params, 'local_crop_channel_scale', '(0.5, 1.0)')),
            last_layer_freeze_iters=getattr(params, 'last_layer_freeze_iters', 1250),
        )

    return model