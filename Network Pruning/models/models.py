from .DAM_resnet import get_DAM_model
def get_model(model, method, num_classes, insize, depth):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, model_name
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable model
    """
    if model in ['resnet']:
        net = get_DAM_model(method, num_classes, depth)
    return net