import torch
import librosa
import numpy as np
import torch.nn.functional as F
from hparams import hparams as hps
from utils.util import to_arr, mode


class Loss(torch.nn.Module):
	def __init__(self):
		super(Loss, self).__init__()
		self.d = 2*hps.sigma*hps.sigma
		self.loss = MultiResolutionSTFTLoss(hps.fft_sizes, hps.hop_sizes,
											hps.win_lengths, hps.mel_scales)

	def forward(self, model_output, p_wavs = None, r_wavs = None):
		# zloss
		z, log_s_list, log_w_list = model_output
		log_s_total = 0
		log_w_total = 0
		for i, log_s in enumerate(log_s_list):
			log_s_total += torch.sum(log_s)
			log_w_total += torch.sum(log_w_list[i])
		zloss = torch.sum(z*z)/self.d-log_s_total-log_w_total
		zloss /= (z.size(0)*z.size(1)*z.size(2))
		
		# sloss
		sloss = self.loss(p_wavs, r_wavs) if p_wavs is not None else 0*zloss

		return zloss+sloss, zloss, sloss


class MultiResolutionSTFTLoss(torch.nn.Module):
	# ref: https://github.com/kan-bayashi/ParallelWaveGAN
	"""Multi resolution STFT loss module."""
	def __init__(self,
				 fft_sizes=[1024, 2048, 512],
				 hop_sizes=[120, 240, 50],
				 win_lengths=[600, 1200, 240],
				 mel_scales=[1, 1, 1],
				 window="hann_window"):
		"""Initialize Multi resolution STFT loss module.

		Args:
			fft_sizes (list): List of FFT sizes.
			hop_sizes (list): List of hop sizes.
			win_lengths (list): List of window lengths.
			window (str): Window function type.

		"""
		super(MultiResolutionSTFTLoss, self).__init__()
		assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
		self.stft_losses = torch.nn.ModuleList()
		self.bases = []
		for fs, ss, wl, sc in zip(fft_sizes, hop_sizes, win_lengths, mel_scales):
			self.stft_losses += [STFTLoss(fs, ss, wl, window)]
			b = librosa.filters.mel(hps.sample_rate, fs, n_mels = hps.num_mels*sc, fmax = hps.fmax).T
			self.bases += [mode(torch.Tensor(b))]

	def forward(self, x, y):
		"""Calculate forward propagation.

		Args:
			x (Tensor): Predicted signal (B, T).
			y (Tensor): Groundtruth signal (B, T).

		Returns:
			Tensor: Multi resolution spectral convergence loss value.
			Tensor: Multi resolution log spectral loss value.

		"""
		sc_loss = 0.0
		spec_loss = 0.0
		for f, b in zip(self.stft_losses, self.bases):
			sc_l, spec_l = f(x, y, b)
			sc_loss += sc_l
			spec_loss += spec_l
		sc_loss /= len(self.stft_losses)
		spec_loss /= len(self.stft_losses)

		return sc_loss+spec_loss


class STFTLoss(torch.nn.Module):
	"""STFT loss module."""

	def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
		"""Initialize STFT loss module."""
		super(STFTLoss, self).__init__()
		self.fft_size = fft_size
		self.shift_size = shift_size
		self.win_length = win_length
		self.window = mode(getattr(torch, window)(win_length))

	def forward(self, x, y, b):
		"""Calculate forward propagation.

		Args:
			x (Tensor): Predicted signal (B, T).
			y (Tensor): Groundtruth signal (B, T).
			b (Tensor): Mel basis (fft_size//2+1, num_mels).

		Returns:
			Tensor: Spectral convergence loss value.
			Tensor: Log STFT magnitude loss value.

		"""
		x_mag, x_mel = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, b)
		y_mag, y_mel = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, b)
		sc_loss = spec_loss = 0
		if hps.mag:
			h = x_mag.size(2)*2*hps.fmax//hps.sample_rate if hps.sample_rate >= 2*hps.fmax else x_mag.size(2)
			x_mag_ = x_mag[:, :, :h]
			y_mag_ = y_mag[:, :, :h]
			sc_loss += torch.norm((y_mag_-x_mag_), p = "fro")/torch.norm(y_mag_, p = "fro")
			spec_loss += torch.nn.L1Loss()(torch.log(x_mag_), torch.log(y_mag_))
			if h < x_mag.size(2):
				x_mag_m = x_mag[:, :, h:].mean(1)
				y_mag_m = y_mag[:, :, h:].mean(1)
				sc_loss += torch.norm((y_mag_m-x_mag_m), p = "fro")/torch.norm(y_mag_m, p = "fro")
				spec_loss += torch.nn.L1Loss()(torch.log(x_mag_m), torch.log(y_mag_m))
		if hps.mel:
			sc_loss += torch.norm((y_mel-x_mel), p = "fro")/torch.norm(y_mel, p = "fro")
			spec_loss += torch.nn.L1Loss()(torch.log(x_mel), torch.log(y_mel))
		s = int(hps.mag)+int(hps.mel)
		if s == 0:
			print('Error: hps.mag and hps.mel are both set as False.')
			exit()
		return sc_loss/s, spec_loss/s


def stft(x, fft_size, hop_size, win_length, window, b):
	"""Perform STFT and convert to magnitude spectrogram.

	Args:
		x (Tensor): Input signal tensor (B, T).
		fft_size (int): FFT size.
		hop_size (int): Hop size.
		win_length (int): Window length.
		window (str): Window function type.
		b (Tensor): Mel basis (fft_size//2+1, num_mels).

	Returns:
		Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

	"""
	x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
	real = x_stft[..., 0]
	imag = x_stft[..., 1]

	# NOTE(kan-bayashi): clamp is needed to avoid nan or inf
	mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
	return mag, torch.clamp(torch.matmul(mag, b), min = 1e-7**0.5)

