from torch.optim import Optimizer, Adam


def get_adam_optimizer(*models, lr: float = 1e-4, betas: tuple = (0.9, 0.999), delta: float = 1e-8) -> Optimizer:
    """
    Get Adam optimizer for jointly training ${models} argument.
    :param models: one or more models to apply optimizer on
    :param lr: learning rate
    :param betas: (p_1, p_2) exponential decay parameters for Adam optimiser's moment estimates. According to
    Goodfellow, good defaults are (0.9, 0.999).
    :param delta: small constant that's used for numerical stability. According to Goodfellow, good defaults is 1e-8.
    :return: instance of torch.optim.Adam optimizer
    """
    joint_params = []
    for model in models:
        joint_params += list(model.parameters())
    return Adam(joint_params, lr=lr, betas=betas, eps=delta)
