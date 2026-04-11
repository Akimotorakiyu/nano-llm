"""
C-AdamW: Cautious AdamW Optimizer
Based on: https://github.com/kyleliang919/C-Optim
"""
from typing import Callable, Tuple, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class CAdamW(Optimizer):
    """
    Cautious AdamW - AdamW with cautious update masking.

    The cautious mask ensures that parameters are only updated in directions
    where the momentum agrees with the gradient direction.

    Arguments:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Weight decay (decoupled, as in AdamW)
                if weight_decay > 0.0:
                    p.mul_(1 - lr * weight_decay)

                # Update momentum and variance
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(eps)

                # Cautious mask: only update where momentum agrees with gradient
                # mask = (exp_avg * grad > 0)
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))

                # Apply cautious update
                step_size = lr / bias_correction1
                denom = denom / (bias_correction2 ** 0.5)

                p.add_(exp_avg * mask / denom, alpha=-step_size)

        return loss
