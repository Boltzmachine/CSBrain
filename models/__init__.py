def _count_bands(band_cutoffs):
    """K = (#cut frequencies) + 1. Accepts a CSV string or iterable."""
    if isinstance(band_cutoffs, str):
        cuts = [c for c in band_cutoffs.split(',') if c.strip()]
        return len(cuts) + 1
    return len(list(band_cutoffs)) + 1


def _spectral_band_kwargs(params):
    """Collect the learnable-filterbank / cross-band / multi-level-alignment
    options for CSBrainAlign, all defaulting to the feature being OFF."""
    return dict(
        use_spectral_bands=getattr(params, 'use_spectral_bands', False),
        num_spectral_bands=getattr(params, 'num_spectral_bands', 4),
        fs=getattr(params, 'fs', 200),
        filterbank_kernel_size=getattr(params, 'filterbank_kernel_size', 101),
        filterbank_min_bw_hz=getattr(params, 'filterbank_min_bw_hz', 1.0),
        use_cross_band_attn=getattr(params, 'use_cross_band_attn', True),
        cross_band_every=getattr(params, 'cross_band_every', 1),
        use_band_type_embedding=getattr(params, 'use_band_type_embedding', True),
        num_visual_levels=getattr(params, 'num_visual_levels', 3),
        band_decorr_weight=getattr(params, 'band_decorr_weight', 0.01),
    )


def _load_eeg_ckpt(encoder, ckpt_path):
    """Warm-start a ``CSBrainAlign`` EEG backbone from a pretraining
    checkpoint (e.g. masked-patch reconstruction).

    Mirrors the prefix-normalisation in ``models/model_for_physio.py``: drops
    the ``module.`` (DataParallel) prefix and, for WorldModel /
    ActionWorldModel wrapper checkpoints, the wrapper's ``encoder.`` / ``eeg.``
    prefix (detected via the ``TemEmbedEEGLayer`` sentinel, which only appears
    under a wrapper). Then keeps only keys whose name *and shape* match the
    target backbone — this forgivingly skips the frozen vision encoder and
    alignment-only heads (whose dims depend on the vision model) so a DINOv2
    recon checkpoint can warm-start a V-JEPA 2 ActionWorldModel encoder.
    """
    if not ckpt_path:
        return
    from utils.util import load_pretrain_checkpoint
    state_dict, _ = load_pretrain_checkpoint(ckpt_path)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    for prefix in ('encoder.', 'eeg.'):
        if any(k.startswith(prefix + 'TemEmbedEEGLayer.') for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()
                          if k.startswith(prefix)}
            break
    model_sd = encoder.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    encoder.load_state_dict(filtered, strict=False)
    print(f"[ActionWorldModel] warm-start from {ckpt_path}: loaded "
          f"{len(filtered)}/{len(model_sd)} backbone tensors "
          f"({len(state_dict) - len(filtered)} ckpt keys skipped)")
    if len(filtered) < 0.3 * len(model_sd):
        print("[ActionWorldModel] WARNING: <30% of the backbone matched the "
              "checkpoint — check that --eeg_ckpt encoder arch args agree.")


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
            freeze_source_projector=getattr(params, 'freeze_source_projector', False),
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
            use_llm_vq=getattr(params, 'use_llm_vq', False),
            num_language_tokens=getattr(params, 'num_language_tokens', 8),
            max_llm_codebook_size=getattr(params, 'max_llm_codebook_size', 4096),
            llm_vq_aux_weight=getattr(params, 'llm_vq_aux_weight', 0.1),
            spectral_mode=getattr(params, 'spectral_mode', 'static'),
            stft_n_fft=getattr(params, 'stft_n_fft', 64),
            stft_hop=getattr(params, 'stft_hop', 1),
            use_moe=getattr(params, 'use_moe', False),
            num_experts=getattr(params, 'num_experts', 4),
            moe_top_k=getattr(params, 'moe_top_k', 2),
            moe_gate_input_dim=getattr(params, 'moe_gate_input_dim', 32),
            moe_balance_weight=getattr(params, 'moe_balance_weight', 0.01),
            moe_band_prior_weight=getattr(params, 'moe_band_prior_weight', 0.1),
            moe_z_loss_weight=getattr(params, 'moe_z_loss_weight', 1e-3),
            contrastive_band=getattr(params, 'contrastive_band', False),
            contrastive_n_bands=_count_bands(getattr(params, 'band_cutoffs', '1,10')),
            contrastive_proj_dim=getattr(params, 'contrastive_proj_dim', 64),
            vision_encoder=getattr(params, 'vision_encoder', 'facebook/dinov2-base'),
            image_pool_heads=getattr(params, 'image_pool_heads', 4),
            use_volume_conduction=getattr(params, 'use_volume_conduction', False),
            vc_tau_init=getattr(params, 'vc_tau_init', 0.08),
            lateralization_flip=getattr(params, 'lateralization_flip', False),
            flip_align_weight=getattr(params, 'flip_align_weight', 1.0),
            lat_sparsity_weight=getattr(params, 'lat_sparsity_weight', 0.01),
            flip_split_hidden=getattr(params, 'flip_split_hidden', 64),
            flip_n_col_bands=getattr(params, 'flip_n_col_bands', 2),
            flip_motion_ref=getattr(params, 'flip_motion_ref', 0.0),
            flip_motion_min=getattr(params, 'flip_motion_min', 0.0),
            **_spectral_band_kwargs(params),
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
            freeze_source_projector=getattr(params, 'freeze_source_projector', False),
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
            use_llm_vq=getattr(params, 'use_llm_vq', False),
            num_language_tokens=getattr(params, 'num_language_tokens', 8),
            max_llm_codebook_size=getattr(params, 'max_llm_codebook_size', 4096),
            llm_vq_aux_weight=getattr(params, 'llm_vq_aux_weight', 0.1),
            spectral_mode=getattr(params, 'spectral_mode', 'static'),
            stft_n_fft=getattr(params, 'stft_n_fft', 64),
            stft_hop=getattr(params, 'stft_hop', 1),
            use_moe=getattr(params, 'use_moe', False),
            num_experts=getattr(params, 'num_experts', 4),
            moe_top_k=getattr(params, 'moe_top_k', 2),
            moe_gate_input_dim=getattr(params, 'moe_gate_input_dim', 32),
            moe_balance_weight=getattr(params, 'moe_balance_weight', 0.01),
            moe_band_prior_weight=getattr(params, 'moe_band_prior_weight', 0.1),
            moe_z_loss_weight=getattr(params, 'moe_z_loss_weight', 1e-3),
            vision_encoder=getattr(params, 'vision_encoder', 'facebook/dinov2-base'),
            image_pool_heads=getattr(params, 'image_pool_heads', 4),
            use_volume_conduction=getattr(params, 'use_volume_conduction', False),
            vc_tau_init=getattr(params, 'vc_tau_init', 0.08),
            lateralization_flip=getattr(params, 'lateralization_flip', False),
            flip_align_weight=getattr(params, 'flip_align_weight', 1.0),
            lat_sparsity_weight=getattr(params, 'lat_sparsity_weight', 0.01),
            flip_split_hidden=getattr(params, 'flip_split_hidden', 64),
            flip_n_col_bands=getattr(params, 'flip_n_col_bands', 2),
            flip_motion_ref=getattr(params, 'flip_motion_ref', 0.0),
            flip_motion_min=getattr(params, 'flip_motion_min', 0.0),
            **_spectral_band_kwargs(params),
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
            target_momentum=getattr(params, 'target_momentum', 0.998),
            flip_pred_weight=getattr(params, 'flip_pred_weight', 1.0),
        )
        return model
    elif params.model == 'ActionWorldModel':
        # Action-conditioned world model: EEG = action, frozen V-JEPA 2 =
        # world. See models/action_world_model.py and plans/world_model.md.
        from .alignment import CSBrainAlign
        from .action_world_model import (
            ActionHead, ActionPredictor, ActionWorldModelWrapper,
        )
        _mbp = getattr(params, 'mamba_band_periods', None)
        if isinstance(_mbp, str) and _mbp:
            _mbp = eval(_mbp)
        # EEG backbone. Alignment / equivariance / info-max are OFF — this
        # paradigm has no image-contrastive term; the EEG net only decodes the
        # action latent. The frozen V-JEPA 2 lives inside as
        # ``pretrained_image_encoder`` and supplies the world states.
        eeg = CSBrainAlign(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward,
            params.seq_len, params.n_layer, params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            causal=getattr(params, 'causal', False),
            project_to_source=getattr(params, 'project_to_source', False),
            num_sources=getattr(params, 'num_sources', 32),
            source_projector_ckpt=getattr(params, 'source_projector_ckpt', None),
            freeze_source_projector=getattr(params, 'freeze_source_projector', False),
            adversarial_weight=0.0,
            equivariance_weight=0.0,
            info_max_weight=0.0,
            alignment_weight=0.0,
            patch_embed_type=getattr(params, 'patch_embed_type', 'cnn'),
            mamba_band_periods=_mbp,
            n_mamba_layers=getattr(params, 'n_mamba_layers', 2),
            mamba_d_state=getattr(params, 'mamba_d_state', 16),
            mamba_d_conv=getattr(params, 'mamba_d_conv', 4),
            mamba_expand=getattr(params, 'mamba_expand', 2),
            use_llm_vq=getattr(params, 'use_llm_vq', False),
            spectral_mode=getattr(params, 'spectral_mode', 'static'),
            stft_n_fft=getattr(params, 'stft_n_fft', 64),
            stft_hop=getattr(params, 'stft_hop', 1),
            use_moe=getattr(params, 'use_moe', False),
            num_experts=getattr(params, 'num_experts', 4),
            moe_top_k=getattr(params, 'moe_top_k', 2),
            moe_gate_input_dim=getattr(params, 'moe_gate_input_dim', 32),
            moe_balance_weight=getattr(params, 'moe_balance_weight', 0.01),
            moe_band_prior_weight=getattr(params, 'moe_band_prior_weight', 0.1),
            moe_z_loss_weight=getattr(params, 'moe_z_loss_weight', 1e-3),
            vision_encoder=getattr(
                params, 'vision_encoder', 'facebook/vjepa2-vitl-fpc64-256'),
            image_pool_heads=getattr(params, 'image_pool_heads', 4),
            use_volume_conduction=getattr(params, 'use_volume_conduction', False),
            vc_tau_init=getattr(params, 'vc_tau_init', 0.08),
            **_spectral_band_kwargs(params),
        )
        _load_eeg_ckpt(eeg, getattr(params, 'eeg_ckpt', None))

        max_horizon = getattr(params, 'max_horizon', 1)
        action_dim = getattr(params, 'action_dim', 64)
        action_head = ActionHead(
            d_model=params.d_model,
            action_dim=action_dim,
            n_action_tokens=getattr(params, 'n_action_tokens', 8),
            n_heads=getattr(params, 'action_pool_heads', 4),
        )
        predictor = ActionPredictor(
            world_dim=eeg.image_feature_dim,
            action_dim=action_dim,
            predictor_d_model=getattr(params, 'predictor_d_model', 512),
            n_layers=getattr(params, 'predictor_n_layers', 4),
            n_heads=getattr(params, 'predictor_n_heads', 8),
            dim_feedforward=getattr(params, 'predictor_dim_feedforward', 1024),
            dropout=getattr(params, 'dropout', 0.1),
            max_horizon=max(max_horizon, 1),
            pred_residual=bool(getattr(params, 'pred_residual', False)),
        )
        model = ActionWorldModelWrapper(
            eeg=eeg,
            predictor=predictor,
            action_head=action_head,
            pred_l1_weight=getattr(params, 'pred_l1_weight', 1.0),
            pred_cos_weight=getattr(params, 'pred_cos_weight', 0.0),
            max_horizon=max_horizon,
            recon_aux_weight=getattr(params, 'recon_aux_weight', 0.0),
            recon_mask_ratio=getattr(params, 'mask_ratio', 0.5),
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
            vision_encoder=getattr(params, 'vision_encoder', 'facebook/dinov2-base'),
            use_saliency=getattr(params, 'use_saliency', False),
            saliency_rollout_skip_layers=getattr(params, 'saliency_rollout_skip_layers', 2),
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