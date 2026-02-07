from .sam import SAM
import numpy as np
import torch
import time

class SLSAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, s=0.5, **kwargs):
        self.s = s
        self.prob = None
        self.alpha_p = 0.001
        self.indx_list = []
        super().__init__(params=params, base_optimizer=base_optimizer, rho=rho, **kwargs)

        self.init_distribution()
        self.set_requires_grad()

    def init_distribution(self):
        len_param = sum(len(sub_group["params"]) for sub_group in self.param_groups)
        self.prob = self.s * np.ones(len_param)


    def set_requires_grad(self):
        self.indx_list = []
        while True:
            num_req_grad = 0
            idx = 0
            for k, group in enumerate(self.param_groups):
                for i, p in enumerate(group["params"]):
                    if self.prob is None:
                        threshold = 0.5
                    else:
                        threshold = self.prob[idx]
                    if np.random.rand() > threshold:
                        p.requires_grad_(False)
                    else:
                        num_req_grad += 1
                        p.requires_grad_(True)
                        self.indx_list.append(idx)
                    idx += 1
            if num_req_grad > 0:
                break

    def get_prob(self, abs_grad_array, idx_array):
        L = max(abs_grad_array)
        p_min = 0.001        
        w = self.prob
        budget = self.s * len(w)
        prob_array = self.prob[idx_array]
        l_hat_raw = - (abs_grad_array / prob_array) ** 2 + (L / p_min) ** 2
        l_hat = l_hat_raw - np.min(l_hat_raw)
        w[idx_array] = prob_array * np.exp(-self.alpha_p * l_hat / prob_array)
        sort_indx = np.argsort(-w)
        w_sort = -np.sort(-w)
        j = 0 
        while (j < len(w)):
            i = sort_indx[j]
            temp_sum = max(sum(w_sort[j:]), 1e-5)
            if w[i] * budget <= temp_sum:
                for k in range(j, len(w)):
                    i = sort_indx[k]
                    self.prob[i] = w[i] * budget / float(temp_sum)
                j = len(w) + 1
            else:
                self.prob[i] = 1
                budget -= 1
            j += 1
        self.prob = np.clip(self.prob, a_min=1e-8, a_max=1.0)


    def update_distribution(self):
        self.get_prob(self.first_grad_norm, np.array(self.indx_list))


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm, self.first_grad_norm = self._grad_norm()
        for k, group in enumerate(self.param_groups):
            scale = group["rho"] / (grad_norm + 1e-12)
            for i, p in enumerate(group["params"]):
                self.state[p]["old_p"] = p.data.clone()
                if p.grad is None: 
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

        
    @torch.no_grad()
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
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        self.update_distribution()
        self.set_requires_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grad_norm_tensor = torch.stack([
            (p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None])
        norm = torch.norm(grad_norm_tensor, p=2)
        grad_norm_array = np.array(grad_norm_tensor.float().cpu())
        return norm, grad_norm_array


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups