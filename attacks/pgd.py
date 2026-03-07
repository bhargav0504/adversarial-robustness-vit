import torchattacks


def get_pgd_attack(model, config):
    """Projected Gradient Descent attack."""
    eps   = config['attacks']['pgd']['eps']
    alpha = config['attacks']['pgd']['alpha']
    steps = config['attacks']['pgd']['steps']
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
