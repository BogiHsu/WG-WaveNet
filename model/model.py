import torch
from utils.util import mode
from model.module import WN, US
from hparams import hparams as hps
from model.layer import Invertible1x1Conv
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		assert(hps.n_group % 2 == 0)
		self.n_group = hps.n_group
		self.n_flows = hps.n_flows
		self.upsample = US()
		self.WN = WN(int(hps.n_group/2), hps.num_mels*hps.n_group)
		self.convinv = torch.nn.ModuleList()
		for k in range(hps.n_flows):
			self.convinv.append(Invertible1x1Conv(hps.n_group))
		self.PF = WN(1, hps.num_mels, True)

	def forward(self, wavs, mels):
		'''
		wavs: (batch_size, seg_l)
		mels: (batch_size, num_mels, T)
		'''
		#  Upsample spectrogram to size of audio
		mels = self.upsample(mels)
		assert(mels.size(2) == wavs.size(1))

		mels = mels.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) # (batch_size, seg_l//n_group, num_mels, n_group)
		mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1) # (batch_size, num_mels*n_group, seg_l//n_group)

		audio = wavs.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # (batch_size, n_group, seg_l//n_group)

		log_s_list = []
		log_det_W_list = []

		for k in range(self.n_flows):
			audio, log_det_W = self.convinv[k](audio)
			log_det_W_list.append(log_det_W)

			n_half = int(audio.size(1)/2)
			audio_0 = audio[:,:n_half,:]
			audio_1 = audio[:,n_half:,:]

			output = self.WN(audio_0, mels).clamp(-10, 10)
			log_s = output[:, n_half:, :]
			b = output[:, :n_half, :]
			audio_1 = torch.exp(log_s)*audio_1+b
			log_s_list.append(log_s)

			audio = torch.cat([audio_0, audio_1],1).clamp(-10, 10)
		
		return audio, log_s_list, log_det_W_list

	def WG(self, inp_mels):
		'''
		mels: (batch_size, num_mels, T)
		'''
		# (batch_size, T//n_group, num_mels, n_group)
		mels = inp_mels.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
		# (batch_size, num_mels*n_group, T//n_group)
		mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1)

		audio = torch.FloatTensor(mels.size(0),
								hps.n_group,
								mels.size(2)).normal_()

		audio = mode(hps.sigma*audio)

		for k in reversed(range(self.n_flows)):
			n_half = int(audio.size(1)/2)
			audio_0 = audio[:,:n_half,:]
			audio_1 = audio[:,n_half:,:]

			output = self.WN(audio_0, mels)
			s = output[:, n_half:, :]
			b = output[:, :n_half, :]
			audio_1 = (audio_1 - b)/torch.exp(s)
			audio = torch.cat([audio_0, audio_1],1)

			audio = self.convinv[k](audio, reverse = True)

		audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), 1, -1) # (batch_size, 1, seg_l)
		return audio

	def infer(self, mels):
		'''
		mels: (batch_size, num_mels, T')
		'''
		inp_mels = self.upsample(mels) # (batch_size, num_mels, T)
		audio = self.WG(inp_mels)
		d = inp_mels.size(2)-audio.size(2)
		if d > 0:
			audio = torch.cat([audio, 0*audio[:, :, :d]], 2)
		audio = self.PF(audio, inp_mels).squeeze(1)
		return audio

	def set_inverse(self):
		for i in range(hps.n_flows):
			self.convinv[i].set_inverse()
	
	@staticmethod
	def remove_weightnorm(model):
		waveglow = model
		for WN in [waveglow.WN, waveglow.PF]:
			WN.start = torch.nn.utils.remove_weight_norm(WN.start)
			WN.in_layers = remove(WN.in_layers)
			WN.cond_layers = remove(WN.cond_layers)
			WN.res_skip_layers = remove(WN.res_skip_layers)
		return waveglow


def remove(conv_list):
	new_conv_list = torch.nn.ModuleList()
	for old_conv in conv_list:
		old_conv = torch.nn.utils.remove_weight_norm(old_conv)
		new_conv_list.append(old_conv)
	return new_conv_list
