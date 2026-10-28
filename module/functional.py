import math
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..q_cost_config import q_cost as default_q_cost


def merge_parameters(base, override):
    result = deepcopy(dict(base))

    if not override:
        override = {}
    else:
        override = dict(override)

    for key in override:
        value = override[key]

        if key not in result:
            raise KeyError(f'Unknown Q-cost parameter: {key}')

        current = result[key]

        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise TypeError(f'{key} must be a dictionary')
            merged_child = merge_parameters(current, value)
            result[key] = merged_child
            continue

        copied_value = deepcopy(value)
        result[key] = copied_value

    return result


def resolve_parameters(override=None):
    cfg = merge_parameters(default_q_cost, override)

    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
            return

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Q-cost scalar parameters must be finite')

    check_finite(cfg)

    domain_names = cfg['domain_names']
    domain_modalities = cfg['domain_modalities']

    if not domain_names:
        raise ValueError(
            'Domain names must be unique and match domain modalities')
    if len(domain_names) != len(domain_modalities):
        raise ValueError(
            'Domain names must be unique and match domain modalities')
    if len(set(domain_names)) != len(domain_names):
        raise ValueError(
            'Domain names must be unique and match domain modalities')

    modality_names = cfg['modality_names']
    if not modality_names:
        raise ValueError('Domain modality indices are out of range')

    for modality in domain_modalities:
        if not isinstance(modality, int):
            raise ValueError('Domain modality indices are out of range')
        if modality < 0:
            raise ValueError('Domain modality indices are out of range')
        if modality >= len(modality_names):
            raise ValueError('Domain modality indices are out of range')

    domain_id = cfg['domain_id']
    epsilon = cfg['epsilon']

    if domain_id < 0:
        raise ValueError('Invalid initial domain or numerical epsilon')
    if domain_id >= len(domain_names):
        raise ValueError('Invalid initial domain or numerical epsilon')
    if epsilon <= 0:
        raise ValueError('Invalid initial domain or numerical epsilon')

    for key in ('prototype_momentum',):
        momentum = cfg['pfha'][key]
        if momentum < 0:
            raise ValueError(f'{key} must lie in [0, 1)')
        if momentum >= 1:
            raise ValueError(f'{key} must lie in [0, 1)')

    activity_momentum = cfg['dqar']['activity_momentum']
    if activity_momentum < 0:
        raise ValueError('Activity momentum must lie in [0, 1)')
    if activity_momentum >= 1:
        raise ValueError('Activity momentum must lie in [0, 1)')

    routing_temperature = cfg['pfha']['routing_temperature']
    similarity_temperature = cfg['dqar']['similarity_temperature']
    if routing_temperature <= 0:
        raise ValueError('Similarity temperatures must be positive')
    if similarity_temperature <= 0:
        raise ValueError('Similarity temperatures must be positive')

    queue_length = cfg['dqar']['queue_length']
    if queue_length < 2:
        raise ValueError(
            'queue_length must include at least one past response')

    warmup_responses = cfg['dqar']['warmup_responses']
    if warmup_responses < 1:
        raise ValueError('warmup_responses must fit the response queue')
    if warmup_responses >= queue_length:
        raise ValueError('warmup_responses must fit the response queue')

    normalization = cfg['dqar']['activity_normalization']
    if normalization not in ('l2', 'minmax', 'max'):
        raise ValueError('Unsupported activity normalization')

    loss_location = cfg['dcpe']['loss_location']
    if loss_location not in ('input', 'output'):
        raise ValueError('DCPE loss_location must be input or output')

    for key in ('activity_decay', 'promotion', 'inhibition', 'sensitivity',
                'amplitude', 'loss_weight'):
        if cfg['dqar'][key] < 0:
            raise ValueError(f'{key} must be nonnegative')

    if cfg['pfha']['loss_weight'] < 0:
        raise ValueError('Auxiliary loss weights must be nonnegative')
    if cfg['dcpe']['loss_weight'] < 0:
        raise ValueError('Auxiliary loss weights must be nonnegative')

    if cfg['pfha']['modality_loss_weight'] < 0:
        raise ValueError(
            'PFHA alignment and residual scales must be nonnegative')
    if cfg['pfha']['residual_scale'] < 0:
        raise ValueError(
            'PFHA alignment and residual scales must be nonnegative')

    # breakpoint()
    return cfg


def unit(value, epsilon):
    value = value.float()
    normalized = F.normalize(value, p=2, dim=-1, eps=epsilon)
    return normalized


def distributed_mean(value, synchronize=True):
    result = value.detach()
    result = result.float()
    result = result.clone()

    if not synchronize:
        return result

    if not dist.is_available():
        return result

    if not dist.is_initialized():
        return result

    dist.all_reduce(result)
    world_size = dist.get_world_size()
    result.div_(world_size)

    return result


def pooled_features(feature, mask=None):
    if feature.ndim != 4:
        raise ValueError('Expected a B,C,H,W feature map')

    if mask is None:
        pooled = feature.float().mean(dim=(-2, -1))
        return pooled

    batch = feature.shape[0]
    height = feature.shape[-2]
    width = feature.shape[-1]
    expected_shape = (batch, height, width)

    if mask.shape != expected_shape:
        raise ValueError('Feature padding mask has an incompatible shape')

    valid = (~mask).float()
    valid = valid.unsqueeze(1)

    weighted = feature.float() * valid
    numerator = weighted.sum(dim=(-2, -1))

    denominator = valid.sum(dim=(-2, -1))
    denominator = denominator.clamp_min(1)

    pooled = numerator / denominator
    return pooled


def configure_runner(cfg, parameters, domain_id=None):
    dino = parameters['dino']
    runtime = parameters['runtime']

    q_cost = resolve_parameters(parameters['q_cost'])

    if domain_id is not None:
        q_cost['domain_id'] = domain_id
        q_cost = resolve_parameters(q_cost)

    #breakpoint()

    depth = dino['backbone_depth']
    supported_depths = (18, 34, 50, 101, 152)

    if depth not in supported_depths:
        raise ValueError('Unsupported ResNet depth')

    hidden_dim = dino['hidden_dim']
    num_queries = dino['num_queries']
    num_classes = dino['num_classes']
    num_heads = dino['num_heads']
    encoder_layers = dino['encoder_layers']
    decoder_layers = dino['decoder_layers']

    smallest_dim = min(
        hidden_dim,
        num_queries,
        num_classes,
        num_heads,
        encoder_layers,
        decoder_layers,
    )
    if smallest_dim < 1:
        raise ValueError('Model dimensions and layer counts must be positive')

    if hidden_dim % 4 != 0:
        raise ValueError(
            'hidden_dim must be divisible by four and by num_heads')
    if hidden_dim % num_heads != 0:
        raise ValueError(
            'hidden_dim must be divisible by four and by num_heads')

    batch_size = runtime['batch_size']
    max_epochs = runtime['max_epochs']
    accumulate_steps = runtime['accumulate_steps']

    if batch_size < 1 or max_epochs < 1 or accumulate_steps < 1:
        raise ValueError(
            'Batch size, epoch count and accumulation must be positive')

    dropout = dino['dropout']
    label_noise_scale = dino['label_noise_scale']
    box_noise_scale = dino['box_noise_scale']

    if not 0 <= dropout < 1:
        raise ValueError('Invalid DINO dropout or denoising strength')
    if not 0 <= label_noise_scale <= 1:
        raise ValueError('Invalid DINO dropout or denoising strength')
    if box_noise_scale < 0:
        raise ValueError('Invalid DINO dropout or denoising strength')

    learning_rate = runtime['learning_rate']
    backbone_lr_mult = runtime['backbone_lr_mult']
    weight_decay = runtime['weight_decay']

    if learning_rate <= 0 or backbone_lr_mult < 0 or weight_decay < 0:
        raise ValueError('Invalid optimizer settings')

    clip_grad_norm = runtime['clip_grad_norm']
    if clip_grad_norm <= 0:
        raise ValueError(
            'Invalid gradient clipping norm or learning-rate milestones')

    for step in runtime['lr_milestones']:
        step_in_range = 0 < step < max_epochs
        if not step_in_range:
            raise ValueError(
                'Invalid gradient clipping norm or learning-rate milestones')

    model = cfg.model
    if model.type != 'DINO':
        raise ValueError(
            'Q-cost requires an MMDetection DINO configuration')

    model.q_cost_cfg = q_cost
    model.num_queries = num_queries
    model.num_feature_levels = dino['num_feature_levels']
    model.backbone.depth = depth
    model.backbone.frozen_stages = dino['backbone_frozen_stages']

    if depth < 50:
        channels = (64, 128, 256, 512)
    else:
        channels = (256, 512, 1024, 2048)

    neck_channels = []
    for index in model.backbone.out_indices:
        neck_channels.append(channels[index])
    model.neck.in_channels = neck_channels

    num_feature_levels = dino['num_feature_levels']
    if num_feature_levels < len(model.neck.in_channels):
        raise ValueError(
            'num_feature_levels must cover all selected backbone outputs')

    if dino['backbone_checkpoint']:
        init_cfg = dict(
            type='Pretrained',
            checkpoint=dino['backbone_checkpoint'],
        )
        model.backbone.init_cfg = init_cfg
    else:
        model.backbone.init_cfg = None

    model.neck.out_channels = hidden_dim
    model.neck.num_outs = num_feature_levels

    neck_groups = model.neck.norm_cfg.get('num_groups', 1)
    if hidden_dim % neck_groups != 0:
        raise ValueError(
            'hidden_dim must be divisible by neck normalization groups')

    model.positional_encoding.num_feats = hidden_dim // 2
    model.positional_encoding.temperature = dino['positional_temperature']

    coders = (
        (model.encoder, 'encoder_layers'),
        (model.decoder, 'decoder_layers'),
    )
    for coder, key in coders:
        coder.num_layers = dino[key]

        layer = coder.layer_cfg
        layer.self_attn_cfg.embed_dims = hidden_dim
        layer.self_attn_cfg.num_heads = num_heads
        layer.self_attn_cfg.dropout = dropout

        layer.ffn_cfg.embed_dims = hidden_dim
        layer.ffn_cfg.feedforward_channels = dino['feedforward_channels']
        layer.ffn_cfg.ffn_drop = dropout

    model.encoder.layer_cfg.self_attn_cfg.num_levels = num_feature_levels

    cross_attn_cfg = model.decoder.layer_cfg.cross_attn_cfg
    cross_attn_cfg.update(
        embed_dims=hidden_dim,
        num_heads=num_heads,
        num_levels=num_feature_levels,
        dropout=dropout,
    )

    model.bbox_head.num_classes = num_classes
    model.bbox_head.loss_cls.loss_weight = dino['classification_weight']
    model.bbox_head.loss_bbox.loss_weight = dino['bbox_weight']
    model.bbox_head.loss_iou.loss_weight = dino['giou_weight']
    model.test_cfg.max_per_img = dino['max_per_img']

    model.dn_cfg.label_noise_scale = label_noise_scale
    model.dn_cfg.box_noise_scale = box_noise_scale
    model.dn_cfg.group_cfg.num_dn_queries = dino['num_dn_queries']

    cfg.optim_wrapper.update(
        type='OptimWrapper',
        accumulative_counts=accumulate_steps,
    )
    cfg.optim_wrapper.optimizer.update(
        type='AdamW',
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    backbone_keys = dict(lr_mult=backbone_lr_mult)
    custom_keys = {'backbone': backbone_keys}
    cfg.optim_wrapper.paramwise_cfg = dict(custom_keys=custom_keys)
    cfg.optim_wrapper.clip_grad = dict(
        max_norm=clip_grad_norm,
        norm_type=2,
    )

    cfg.train_cfg.update(
        max_epochs=max_epochs,
        val_interval=runtime['val_interval'],
    )

    scheduler = dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=list(runtime['lr_milestones']),
        gamma=runtime['lr_gamma'],
    )
    cfg.param_scheduler = [scheduler]

    #breakpoint()

    cfg.train_dataloader.batch_size = batch_size
    cfg.randomness = dict(seed=runtime['seed'], deterministic=False)
    cfg.find_unused_parameters = runtime['find_unused_parameters']
    cfg.auto_scale_lr = dict(
        enable=False,
        base_batch_size=batch_size * 2,
    )

    hooks = []
    existing_hooks = cfg.get('custom_hooks', [])
    for hook in existing_hooks:
        if hook['type'] != 'QCostIncrementalHook':
            hooks.append(hook)

    if q_cost['enabled']:
        hooks.append(dict(
            type='QCostIncrementalHook',
            domain_id=q_cost['domain_id'],
        ))

    cfg.custom_hooks = hooks

# breakpoint()
    return cfg
