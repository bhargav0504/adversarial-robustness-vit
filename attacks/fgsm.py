import torchattacks


def get_fgsm_attack(model, config):
    eps = config['attacks']['fgsm']['eps']
    return torchattacks.FGSM(model, eps=eps)
