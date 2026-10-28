import torch
from torch import nn

from .functional import unit
from .query_activity_buffer import QueryActivityBuffer


class DomainAwareQueryActivityRegulator(nn.Module):
    def __init__(self, num_queries, embed_dims, cfg):
        super().__init__()

        self.cfg = cfg['dqar']
        self.epsilon = cfg['epsilon']
        self.synchronize = cfg['synchronize']

        num_domains = len(cfg['domain_names'])
        self.buffer = QueryActivityBuffer(
            num_queries, embed_dims, num_domains, cfg)

        self.register_buffer(
            'old_queries', torch.zeros(num_queries, embed_dims))
        self.register_buffer('snapshot_valid', torch.tensor(False))

        self.register_buffer(
            'gate', torch.ones(num_queries), persistent=False)

        self.pending_gate = None
        self.pending_responses = []

    @torch.no_grad()
    def begin_domain(self, queries, domain):
        queries = queries.detach().float()
        self.old_queries.copy_(queries)

        has_snapshot = domain > 0
        self.snapshot_valid.fill_(has_snapshot)

        self.buffer.clear_queue()
        self.gate.fill_(1)

        self.pending_gate = None
        self.pending_responses.clear()

        # breakpoint()

    def normalize_activity(self, activity):
        mode = self.cfg['activity_normalization']

        if mode == 'l2':
            normalized = unit(activity, self.epsilon)
            return normalized

        values = activity

        if mode == 'minmax':
            minimum = values.amin(dim=-1, keepdim=True)
            values = values - minimum

        maximum = values.amax(dim=-1, keepdim=True)
        maximum = maximum.clamp_min(self.epsilon)

        normalized = values / maximum
        return normalized

    @torch.no_grad()
    def commit_response(self):
        for response in self.pending_responses:
            self.buffer.push(response)

        self.pending_responses.clear()

        # breakpoint()

    def gate_gradient(self, gradient):
        if self.pending_gate is None:
            return gradient

        gate = self.pending_gate.to(gradient)
        gate = gate[:, None]

        scaled = gradient * gate
        return scaled
