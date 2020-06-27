import torch
import pprint
import logging
import datetime


LOG_FORMAT = '%(asctime)s - %(filename)-20s - %(levelname)-10s - %(message)s'


def get_logger(name=None, fmt=LOG_FORMAT, level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt=fmt)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if level:
        logger.setLevel(level)
    return logger


def get_timestamp(fmt='%b%d_%H-%M-%S'):
    return datetime.datetime.now().strftime(fmt)


def save_model(model, path, **kwargs):
    """
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        ...
    }, PATH)
    """
    torch.save({'model_state_dict': model.state_dict(), **kwargs}, path)


def load_model(model, path):
    """
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        model.train()
    """

    print(f"Loading model from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    pprint.pprint({k: v for k, v in checkpoint.items() if k != 'model_state_dict'})
    return model
