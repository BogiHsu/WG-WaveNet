import torch
import numpy as np
from hparams import hparams as hps


def mode(obj, model = False):
	d = torch.device('cuda' if hps.is_cuda else 'cpu')
	return obj.to(d, non_blocking = False if model else hps.pin_mem)

def to_arr(var):
	return var.cpu().detach().numpy().astype(np.float32)
