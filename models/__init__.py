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
        model = CSBrainAlign(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions=None,
            # sorted_indices
        )
    else:
        raise ValueError(f"Unknown model type: {params.model}")
    
    return model