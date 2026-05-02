import torchattacks


def get_pgd_attack(model, config):
    eps   = config['attacks']['pgd']['eps']
    alpha = config['attacks']['pgd']['alpha']
    steps = config['attacks']['pgd']['steps']
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
