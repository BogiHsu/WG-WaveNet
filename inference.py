import torch
import argparse
import numpy as np
from model.model import Model
from hparams import hparams as hps
from utils.util import mode, to_arr
from utils.audio import load_wav, save_wav, melspectrogram


def load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth, map_location = None if hps.is_cuda else 'cpu')
	model = Model()
	model.load_state_dict(ckpt_dict['model'])
	model = model.remove_weightnorm(model)
	model = mode(model, True).eval()
	# pre-run
	model.set_inverse()
	with torch.no_grad():
		res = model.infer(mode(torch.zeros((1, 80, 10))))[0]
	torch.cuda.empty_cache()
	return model


def infer(model, src_pth):
	src = load_wav(src_pth, seg = False)
	mel = melspectrogram(src).astype(np.float32)
	mel = mode(torch.Tensor([mel]))
	with torch.no_grad():
		res = model.infer(mel)[0]
	return [src, to_arr(res)]


def audio(outputs, res_pth):
	src = outputs[0]
	res = outputs[1]
	
	# save audio
	save_wav(src, res_pth+'_src.wav')
	save_wav(res, res_pth+'_res.wav')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--ckpt_pth', type = str, default = '',
						required = True, help = 'path to load checkpoints')
	parser.add_argument('-s', '--src_pth', type = str, default = '',
						required = True, help = 'path to source')
	parser.add_argument('-r', '--res_pth', type = str, default = '',
						required = True, help = 'path to save wavs')

	args = parser.parse_args()
	
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False
	model = load_model(args.ckpt_pth)
	outputs = infer(model, args.src_pth)
	audio(outputs, args.res_pth)
