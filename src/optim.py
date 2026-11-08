"""
Optimisers: rules for updating the model's parameters using their gradients.
Each optimiser's step() is called once per training step, after model.backward().
"""

import numpy as np

class SGD:
    """
    Stochastic gradient descent:  theta <- theta - lr * dtheta
    dtheta is the adjoint of THIS step's batch loss (an unbiased estimate
    of the full-data gradient).
    """

    def __init__(self, model, lr):
        self.model = model   # the model whose parameters this optimiser updates
        self.lr = lr         # learning rate (eta): the step size

    def step(self):
        """Updates every parameter in place, using the gradients from the latest backward."""
        # Fetched fresh each step: every backward creates new gradient arrays
        for name, param, grad in self.model.params_and_grads():
            param -= self.lr * grad   # in place, so the model's own arrays change


class Adam:
    """
    Adam (Kingma & Ba, 2014): momentum + per-parameter step sizes.
        m     <- beta1 m + (1 - beta1) g          first moment  (smoothed gradient)
        v     <- beta2 v + (1 - beta2) g (*) g    second moment (typical squared size)
        m_hat =  m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)    bias correction
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr          # eta: roughly how far each parameter moves per step
        self.beta1 = beta1    # momentum decay: averages about the last 1 / (1 - beta1) = 10 steps
        self.beta2 = beta2    # second-moment decay: about the last 1000 steps
        self.eps = eps        # avoids dividing by zero

        self.t = 0            # step counter, for the bias correction
        self.m = {}           # parameter name -> first moment array  (same shape as the parameter)
        self.v = {}           # parameter name -> second moment array (same shape as the parameter)

    def step(self):
        """Updates every parameter in place, using the gradients from the latest backward."""
        self.t += 1

        # Bias-correction factors: the fraction of the true value each average has reached
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        for name, param, grad in self.model.params_and_grads():
            # First step for this parameter: both averages start at zero
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Exponential moving averages of the gradient and the squared gradient
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad * grad

            # Undo the pull towards zero from starting at m = v = 0
            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2

            # Per-parameter step: m_hat / sqrt(v_hat) is about +-1 when the gradient is consistent
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # in place: the model changes
