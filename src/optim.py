"""
Optimisers: rules for updating the model's parameters using their gradients.
Each optimiser's step() is called once per training step, after model.backward().
"""


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