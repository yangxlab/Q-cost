import torch
import torch.nn.functional as F
from torch import nn


class DomainContextPrompt(nn.Module):
    def __init__(self, embed_dims, cfg):
        super().__init__()

        self.cfg = cfg['dcpe']
        self.epsilon = cfg['epsilon']

        projection_bias = self.cfg['projection_bias']
        self.modality_projection = nn.Linear(
            embed_dims, embed_dims, bias=projection_bias)
        self.domain_projection = nn.Linear(
            embed_dims, embed_dims, bias=projection_bias)
        self.position = nn.Parameter(torch.zeros(1, 1, embed_dims))

    def forward(self, context):
        weight_dtype = self.modality_projection.weight.dtype

        modality = context['modality']
        domain = context['domain']

        modality = modality.to(weight_dtype)
        domain = domain.to(weight_dtype)

        modality_context = self.modality_projection(modality)
        domain_context = self.domain_projection(domain)

        prompt = modality_context + domain_context
        # breakpoint()
        return prompt

    def alignment_loss(self, tokens, prompt, mask=None):
        include_prompt = self.cfg['include_prompt_in_pool']

        if not include_prompt:
            tokens = tokens[:, 1:]
            if mask is not None:
                mask = mask[:, 1:]

        if mask is None:
            pooled = tokens.float().mean(dim=1)
        else:
            valid = (~mask).float()
            valid = valid.unsqueeze(-1)

            weighted_tokens = tokens.float() * valid
            numerator = weighted_tokens.sum(dim=1)

            denominator = valid.sum(dim=1)
            denominator = denominator.clamp_min(1)

            pooled = numerator / denominator

        similarity = F.cosine_similarity(
            pooled, prompt.float(), dim=-1, eps=self.epsilon)

        loss = 1 - similarity
        loss = loss.mean()
        loss = loss * self.cfg['loss_weight']

        # breakpoint()
        return loss

    def bind_attention(self, attention):
        offsets = attention.sampling_offsets

        offsets.q_cost_heads = attention.num_heads
        offsets.q_cost_levels = attention.num_levels
        offsets.q_cost_points = attention.num_points

        offsets.register_forward_hook(self.center_prompt_offsets)

        # breakpoint()

    @staticmethod
    def center_prompt_offsets(module, inputs, output):
        shape = output.shape

        heads = module.q_cost_heads
        levels = module.q_cost_levels
        points = module.q_cost_points

        offsets = output.reshape(*shape[:-1], heads, levels, points, 2)
        offsets = offsets.clone()

        offsets[..., 0, :, :] = 0

        restored = offsets.reshape(shape)
        return restored
