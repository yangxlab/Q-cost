import torch
from torch import nn

from .adapter_expert import BottleneckAdapterExpert
from .functional import pooled_features, unit
from .prototype_bank import DomainPrototypeBank


class PrototypeFeatureHomogenizationAdapter(nn.Module):
    def __init__(self, num_levels, embed_dims, cfg):
        super().__init__()

        self.cfg = cfg['pfha']
        self.epsilon = cfg['epsilon']
        self.num_levels = num_levels
        self.embed_dims = embed_dims

        self.bank = DomainPrototypeBank(num_levels, embed_dims, cfg)

        def build_expert():
            expert = BottleneckAdapterExpert(
                embed_dims,
                self.cfg['bottleneck_dim'],
                self.cfg['activation'],
                self.cfg['dropout'],
                self.cfg['zero_init'],
            )
            return expert

        modality_experts = []
        for _ in range(num_levels):
            level_experts = []
            for _ in cfg['modality_names']:
                level_experts.append(build_expert())
            modality_experts.append(nn.ModuleList(level_experts))
        self.modality_experts = nn.ModuleList(modality_experts)

        domain_experts = []
        for _ in range(num_levels):
            level_experts = []
            for _ in cfg['domain_names']:
                level_experts.append(build_expert())
            domain_experts.append(nn.ModuleList(level_experts))
        self.domain_experts = nn.ModuleList(domain_experts)

    def forward(self, features, masks=None, update=False):
        if len(features) != self.num_levels:
            raise ValueError(
                'PFHA feature level count differs from its configuration')

        for feature in features:
            if feature.ndim != 4:
                raise ValueError(
                    'PFHA requires projected features with a common channel '
                    'dimension')
            if feature.shape[1] != self.embed_dims:
                raise ValueError(
                    'PFHA requires projected features with a common channel '
                    'dimension')

        if masks is None:
            masks = [None] * len(features)
        if len(masks) != len(features):
            raise ValueError(
                'PFHA requires one padding mask per feature level')

        pooled_per_level = []
        for feature, mask in zip(features, masks):
            pooled = pooled_features(feature, mask)
            pooled_per_level.append(pooled)
        pooled = torch.stack(pooled_per_level, dim=1)

        if update:
            self.bank.update(pooled)

        outputs = []
        contexts = []
        centers = []

        loss = pooled.sum() * 0

        routing_temperature = self.cfg['routing_temperature']

        for level in range(len(features)):
            feature = features[level]

            routed, weights, center, fused = self.bank.route(
                pooled[:, level].detach(), level, routing_temperature)

            centers.append(center)
            contexts.append(fused)

            if not self.cfg['enabled']:
                outputs.append(feature)
                continue

            adapted = torch.zeros_like(feature)
            experts = self.modality_experts[level]

            for modality in range(len(experts)):
                expert = experts[modality]
                indices = (routed == modality).nonzero(as_tuple=True)[0]

                if indices.numel() == 0:
                    continue

                selected = feature.index_select(0, indices)
                expert_output = expert(selected)
                updated = selected + expert_output
                adapted = adapted.index_copy(0, indices, updated)

            residual = torch.zeros_like(feature)
            domain_experts = self.domain_experts[level]

            for domain in range(len(domain_experts)):
                expert = domain_experts[domain]
                indices = (weights[:, domain] > 0).nonzero(as_tuple=True)[0]

                if indices.numel() == 0:
                    continue

                selected = adapted.index_select(0, indices)
                expert_output = expert(selected)

                scaling = weights[indices, domain]
                scaling = scaling.to(expert_output.dtype)
                scaling = scaling[:, None, None, None]

                expert_output = expert_output * scaling
                residual = residual.index_add(0, indices, expert_output)

            delta = adapted - feature
            delta = delta + residual

            scaled_delta = self.cfg['residual_scale'] * delta
            output = feature + scaled_delta
            outputs.append(output)

            if not update:
                continue

            domain_id = int(self.bank.current_domain)
            prototype = self.bank.prototypes[domain_id, level]

            normalized_pooled = unit(pooled[:, level], self.epsilon)
            normalized_prototype = unit(prototype, self.epsilon)

            domain_loss = (
                normalized_pooled - normalized_prototype).square().sum(dim=-1)
            domain_loss = domain_loss.mean()
            loss = loss + domain_loss

            modality_loss = (
                normalized_pooled - center).square().sum(dim=-1)
            modality_loss = modality_loss.mean()
            loss = loss + self.cfg['modality_loss_weight'] * modality_loss

            # breakpoint()

        centers_tensor = torch.stack(centers, dim=1)
        contexts_tensor = torch.stack(contexts, dim=1)

        context = {
            'modality': centers_tensor.mean(dim=1),
            'domain': contexts_tensor.mean(dim=1),
        }

        # breakpoint()
        return tuple(outputs), context, loss * self.cfg['loss_weight']
