import os
import time
import torch
import argparse
import numpy as np
from utils.util import mode
from model.loss import Loss
from model.model import Model
from utils.logger import Logger
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.dataset import ljdataset, collate_fn
np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)


def prepare_dataloaders(fdir):
	trainset = ljdataset(fdir)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return train_loader


def load_checkpoint(ckpt_pth, model, optimizer):
	ckpt_dict = torch.load(ckpt_pth)
	model.load_state_dict(ckpt_dict['model'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)

def train(args):
	# build model
	model = Model()
	#print(sum(p.numel() for p in model.parameters() if p.requires_grad))
	mode(model, True)
	optimizer = torch.optim.AdamW(model.parameters(), lr = hps.lr)
	criterion = Loss()

	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer)
		iteration += 1 # next iteration is iteration+1
	
	# get scheduler
	if hps.sch:
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.StepLR(
						optimizer, hps.sch_step, hps.sch_g,
						last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.StepLR(
						optimizer, hps.sch_step, hps.sch_g)
	
	# make dataset
	train_loader = prepare_dataloaders(args.data_dir)
	
	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir):
			os.makedirs(args.log_dir)
			os.chmod(args.log_dir, 0o775)
		logger = Logger(args.log_dir)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
		os.chmod(args.ckpt_dir, 0o775)

	model.train()
	# ================ MAIN TRAINNIG LOOP ===================
	while iteration <= hps.max_iter:
		for batch in train_loader:
			if iteration > hps.max_iter:
				break
			start = time.perf_counter()
			wavs, mels = batch
			wavs = mode(wavs)
			mels = mode(mels)

			# forward
			outputs = model(wavs, mels)
			p_wavs = model.infer(mels) if iteration%hps.n == 0 else None
			
			# loss
			loss = criterion(outputs, p_wavs, wavs)
			
			# zero grad ans backward 
			model.zero_grad()
			loss[0].backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.gn)
			
			# update
			optimizer.step()
			if hps.sch:
				scheduler.step(min(iteration, hps.sch_stop))
			
			# info
			dur = time.perf_counter()-start
			print('Iter: {} Loss(z/s): {:.2e}/{:.2e} GN: {:.2e} {:.1f}s/it'.format(
				iteration, loss[1].item(), loss[2].item(), grad_norm, dur))
			# log
			if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training(loss[1].item(), loss[2].item(), learning_rate, iteration)
			
			# save ckpt
			if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
				ckpt_pth = os.path.join(args.ckpt_dir, 'ckpt_{}'.format(iteration))
				save_checkpoint(model, optimizer, iteration, ckpt_pth)

			# sample
			if args.log_dir != '' and (iteration % hps.iters_per_sample == 0):
				model.eval()
				with torch.no_grad():
					pred = model.infer(mels[:1])
					logger.sample_training(wavs[0], pred[0], iteration)
				model.train()
			
			iteration += 1

	if args.log_dir != '':
		logger.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# path
	parser.add_argument('-d', '--data_dir', type = str, default = 'data',
						help = 'directory to load data')
	parser.add_argument('-l', '--log_dir', type = str, default = 'log',
						help = 'directory to save tensorboard logs')
	parser.add_argument('-cd', '--ckpt_dir', type = str, default = 'ckpt',
						help = 'directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'directory to load checkpoints')

	args = parser.parse_args()

	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	train(args)
