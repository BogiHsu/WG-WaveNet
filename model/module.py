import torch
from hparams import hparams as hps
from model.layer import fused_add_tanh_sigmoid_multiply


class WN(torch.nn.Module):
	def __init__(self, n_in_channels, n_emb_channels, PF = False):
		super(WN, self).__init__()
		assert(hps.kernel_size % 2 == 1)
		assert(hps.n_channels % 2 == 0)
		self.PF = PF
		self.n_layers = hps.PF_n_layers if PF else hps.n_layers
		self.n_channels = hps.PF_n_channels if PF else hps.n_channels
		self.in_layers = torch.nn.ModuleList()
		self.cond_layers = torch.nn.ModuleList()
		self.res_skip_layers = torch.nn.ModuleList()

		start = torch.nn.Conv1d(n_in_channels, self.n_channels, 1)
		start = torch.nn.utils.weight_norm(start, name = 'weight')
		self.start = start

		if PF:
			self.end = torch.nn.Conv1d(self.n_channels, n_in_channels, 1)
		else:
			self.end = torch.nn.Conv1d(self.n_channels, 2*n_in_channels, 1)
			self.end.weight.data.zero_()
			self.end.bias.data.zero_()

		for i in range(self.n_layers):
			dilation = 2**i if PF else 2**i
			padding = int((hps.kernel_size*dilation - dilation)/2)
			in_layer = torch.nn.Conv1d(self.n_channels, 2*self.n_channels, hps.kernel_size,
									   dilation = dilation, padding = padding)
			in_layer = torch.nn.utils.weight_norm(in_layer, name = 'weight')
			self.in_layers.append(in_layer)

			cond_layer = torch.nn.Conv1d(n_emb_channels, 2*self.n_channels, 1)
			cond_layer = torch.nn.utils.weight_norm(cond_layer, name = 'weight')
			self.cond_layers.append(cond_layer)

			# last one is not necessary
			if i < self.n_layers - 1:
				res_skip_channels = 2*self.n_channels
			else:
				res_skip_channels = self.n_channels
			res_skip_layer = torch.nn.Conv1d(self.n_channels, res_skip_channels, 1)
			res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name = 'weight')
			self.res_skip_layers.append(res_skip_layer)

	def forward(self, wavs, mels):
		wavs = self.start(wavs)

		for i in range(self.n_layers):
			acts = fused_add_tanh_sigmoid_multiply(
				self.in_layers[i](wavs),
				self.cond_layers[i](mels),
				torch.IntTensor([self.n_channels]))

			res_skip_acts = self.res_skip_layers[i](acts)
			if i < self.n_layers - 1:
				wavs = res_skip_acts[:,:self.n_channels,:] + wavs
				skip_acts = res_skip_acts[:,self.n_channels:,:]
			else:
				skip_acts = res_skip_acts

			if i == 0:
				output = skip_acts
			else:
				output = skip_acts + output
		return self.end(torch.nn.ReLU()(output) if self.PF else output)

class US(torch.nn.Module):
	def __init__(self):
		super(US, self).__init__()
		self.layers = torch.nn.ModuleList()
		self.layers.append(torch.nn.Conv1d(hps.num_mels, hps.num_mels, kernel_size = 2))
		self.layers.append(torch.nn.ReLU())
		for sf in hps.up_scale:		
			self.layers.append(torch.nn.Upsample(scale_factor = sf))
			self.layers.append(torch.nn.Conv1d(hps.num_mels, hps.num_mels,
											kernel_size = 5, padding = 2))
			self.layers.append(torch.nn.ReLU())

	def forward(self, mels):
		for f in self.layers:
			mels = f(mels)
		return mels
'''
class US(torch.nn.Module):
	def __init__(self):
		super(US, self).__init__()
		self.inconv = torch.nn.Conv1d(hps.num_mels, hps.num_mels, kernel_size = 2, bias = False)
		self.layers = torch.nn.ModuleList()
		for sf in hps.up_scale:
			uscnn = torch.nn.ModuleList()
			uscnn.append(torch.nn.Upsample(scale_factor = sf, mode = 'linear'))
			uscnn.append(torch.nn.Conv2d(1, 1, (1, 5), padding = (0, 2), bias = False))
			self.layers.append(uscnn)

	def forward(self, mels):
		mels = self.inconv(mels)
		for f in self.layers:
			mels = f[0](mels)
			mels = mels.unsqueeze(1)
			mels = f[1](mels)
			mels = mels.squeeze(1)
		return mels
'''
