import torchattacks


def get_fgsm_attack(model, config):
    """Fast Gradient Sign Method attack."""
    eps = config['attacks']['fgsm']['eps']
    return torchattacks.FGSM(model, eps=eps)
