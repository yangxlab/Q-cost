import torch
from torch import nn

from .functional import distributed_mean


class QueryActivityBuffer(nn.Module):
    def __init__(self, num_queries, embed_dims, num_domains, cfg):
        super().__init__()

        dqar_cfg = cfg['dqar']
        self.momentum = dqar_cfg['activity_momentum']
        self.synchronize = cfg['synchronize']

        activity = torch.zeros(num_domains, num_queries)
        self.register_buffer('activity', activity)

        initialized = torch.zeros(num_domains, dtype=torch.bool)
        self.register_buffer('initialized', initialized)

        queue_length = dqar_cfg['queue_length']
        queue = torch.zeros(queue_length - 1, num_queries, embed_dims)
        self.register_buffer('queue', queue)

        queue_size = torch.tensor(0, dtype=torch.long)
        self.register_buffer('queue_size', queue_size)

        queue_cursor = torch.tensor(0, dtype=torch.long)
        self.register_buffer('queue_cursor', queue_cursor)

    @torch.no_grad()
    def clear_queue(self):
        self.queue.zero_()
        self.queue_size.zero_()
        self.queue_cursor.zero_()

        # breakpoint()

    def response_mean(self):
        size = int(self.queue_size)
        if size == 0:
            return None

        valid_entries = self.queue[:size]
        mean_response = valid_entries.mean(dim=0)
        mean_response = mean_response.clone()
        return mean_response

    @torch.no_grad()
    def update_activity(self, domain, activity):
        mean_activity = activity.detach().mean(dim=0)
        value = distributed_mean(mean_activity, self.synchronize)

        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite query activity')

        if bool(self.initialized[domain]):
            self.activity[domain].lerp_(value, 1 - self.momentum)
            return

        self.activity[domain].copy_(value)
        self.initialized[domain] = True

        # breakpoint()

    @torch.no_grad()
    def push(self, response):
        mean_response = response.detach().float().mean(dim=0)
        value = distributed_mean(mean_response, self.synchronize)

        expected_shape = self.queue.shape[1:]
        if value.shape != expected_shape:
            raise ValueError('Invalid query response for the activity queue')
        if not torch.isfinite(value).all():
            raise ValueError('Invalid query response for the activity queue')

        cursor = int(self.queue_cursor)
        self.queue[cursor].copy_(value)

        next_cursor = cursor + 1
        next_cursor = next_cursor % self.queue.shape[0]
        self.queue_cursor.fill_(next_cursor)

        current_size = int(self.queue_size)
        next_size = current_size + 1
        next_size = min(next_size, self.queue.shape[0])
        self.queue_size.fill_(next_size)

        # breakpoint()
