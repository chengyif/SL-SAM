from .sam import SAM
import numpy as np
import torch

class RST(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, s=0.5, **kwargs):
        self.s = s
        super().__init__(params=params, base_optimizer=base_optimizer, rho=rho, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for k, group in enumerate(self.param_groups):
            scale = group["rho"] / (grad_norm + 1e-12)
            for i, p in enumerate(group["params"]):
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    def second_step(self, zero_grad=False):
        for k, group in enumerate(self.param_groups):
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        if np.random.rand() > self.s:
            self.base_optimizer.step()
            self.zero_grad()
        else:
            self.first_step(zero_grad=True)
            closure()
            self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups