from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class QCostIncrementalHook(Hook):
    priority = 'NORMAL'

    def __init__(self, domain_id=0):
        self.domain_id = domain_id

    @staticmethod
    def unwrap_model(model):
        current = model
        while is_model_wrapper(current):
            current = current.module
        return current

    def before_train(self, runner):
        model = self.unwrap_model(runner.model)

        enabled = getattr(model, 'q_cost_enabled', False)
        if not enabled:
            raise RuntimeError(
                'QCostIncrementalHook requires an enabled DINO Q-cost model')

        model.begin_domain(self.domain_id)

        domain_names = model.q_cost_cfg['domain_names']
        domain_name = domain_names[self.domain_id]
        runner.logger.info(f'Q-cost domain {self.domain_id}: {domain_name}')

        # breakpoint()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = self.unwrap_model(runner.model)

        dqar_enabled = model.q_cost_cfg['dqar']['enabled']
        if not dqar_enabled:
            return

        regulator = model.query_regulator
        regulator.commit_response()
        regulator.pending_gate = None

        # breakpoint()
