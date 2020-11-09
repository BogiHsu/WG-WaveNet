import numpy as np
from utils.util import to_arr
from hparams import hparams as hps
from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
	def __init__(self, logdir):
		super(Logger, self).__init__(logdir, flush_secs = 5)

	def log_training(self, zloss, sloss, learning_rate, iteration):
			self.add_scalar('zloss', zloss, iteration)
			self.add_scalar('sloss', sloss, iteration)
			self.add_scalar('learning.rate', learning_rate, iteration)

	def sample_training(self, real, pred, iteration):
			real = to_arr(real)
			pred = to_arr(pred)
			
			# save audio
			real /= max(0.01, np.max(np.abs(real)))
			pred /= max(0.01, np.max(np.abs(pred)))
			self.add_audio('real', real, iteration, hps.sample_rate)
			self.add_audio('pred', pred, iteration, hps.sample_rate)
