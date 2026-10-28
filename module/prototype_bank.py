import torch
from torch import nn

from .functional import distributed_mean, unit


class DomainPrototypeBank(nn.Module):
    def __init__(self, num_levels, embed_dims, cfg):
        super().__init__()

        self.epsilon = cfg['epsilon']
        self.momentum = cfg['pfha']['prototype_momentum']
        self.synchronize = cfg['synchronize']

        self.num_modalities = len(cfg['modality_names'])
        num_domains = len(cfg['domain_names'])

        prototypes = torch.zeros(num_domains, num_levels, embed_dims)
        self.register_buffer('prototypes', prototypes)

        initialized = torch.zeros(num_domains, dtype=torch.bool)
        self.register_buffer('initialized', initialized)

        modalities = torch.tensor(cfg['domain_modalities'], dtype=torch.long)
        self.register_buffer('modalities', modalities)

        current_domain = torch.tensor(-1, dtype=torch.long)
        self.register_buffer('current_domain', current_domain)

    @torch.no_grad()
    def begin_domain(self, domain_id):
        capacity = self.prototypes.shape[0]

        if domain_id < 0:
            raise ValueError('Domain index exceeds configured capacity')
        if domain_id >= capacity:
            raise ValueError('Domain index exceeds configured capacity')

        current = int(self.current_domain)

        if domain_id == current:
            return False

        expected = current + 1
        if domain_id != expected:
            raise ValueError(
                f'Expected next domain {expected}, received {domain_id}; '
                'load its predecessor checkpoint')

        self.current_domain.fill_(domain_id)
        return True

    @torch.no_grad()
    def update(self, pooled):
        domain = int(self.current_domain)
        if domain < 0:
            raise RuntimeError('Call begin_domain before updating prototypes')

        mean_pooled = pooled.detach().mean(dim=0)
        value = distributed_mean(mean_pooled, self.synchronize)

        if not torch.isfinite(value).all():
            raise FloatingPointError(
                'Non-finite backbone features cannot update prototypes')

        if not bool(self.initialized[domain]):
            self.prototypes[domain].copy_(value)
            self.initialized[domain] = True
            # breakpoint()
            return

        self.prototypes[domain].lerp_(value, 1 - self.momentum)

        # breakpoint()

    def centers(self, level):
        values = []
        available = []

        prototype_dim = self.prototypes.shape[-1]

        for modality in range(self.num_modalities):
            mask = self.initialized & (self.modalities == modality)
            has_any = bool(mask.any())
            available.append(has_any)

            if has_any:
                selected = self.prototypes[mask, level]
                value = selected.mean(dim=0)
            else:
                value = self.prototypes.new_zeros(prototype_dim)

            values.append(value)

        stacked = torch.stack(values)
        centers = unit(stacked, self.epsilon)

        available_tensor = torch.tensor(
            available,
            device=self.prototypes.device,
            dtype=torch.bool,
        )
        return centers, available_tensor

    def route(self, pooled, level, temperature):
        any_initialized = bool(self.initialized.any())
        if not any_initialized:
            raise RuntimeError(
                'No learned domain prototypes are available; '
                'load a Q-cost checkpoint for inference')

        centers, available = self.centers(level)

        normalized_pooled = unit(pooled, self.epsilon)
        similarity = normalized_pooled @ centers.t()

        masked_similarity = similarity.masked_fill(
            ~available[None], -torch.inf)
        routed = masked_similarity.argmax(-1)

        candidates = self.initialized[None] & (
            self.modalities[None] == routed[:, None])

        prototype_level = self.prototypes[:, level]
        normalized_prototypes = unit(prototype_level, self.epsilon)
        logits = normalized_pooled @ normalized_prototypes.t()

        scaled_logits = logits / temperature
        weights = scaled_logits.masked_fill(~candidates, -torch.inf)
        weights = weights.softmax(-1)

        fused = weights @ prototype_level.float()

        # breakpoint()

        selected_center = centers[routed]
        return routed, weights, selected_center, fused

    def historical_context(self):
        values = self.prototypes.detach()
        values = values.float()
        context = values.mean(dim=1)
        return context
