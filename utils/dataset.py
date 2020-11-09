import os
import torch
import pickle
import numpy as np
from hparams import hparams as hps
from utils.audio import load_wav, melspectrogram
from torch.utils.data import Dataset, DataLoader


def files_to_list(fdir):
	f_list = []
	with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
			if hps.prep:
				wav = load_wav(wav_path, False)
				if wav.shape[0] < hps.seg_l:
					wav = np.pad(wav, (0, hps.seg_l-wav.shape[0]), 'constant', constant_values = (0, 0))
				mel = melspectrogram(wav).astype(np.float32) 
				f_list.append([wav, mel])
			else:
				f_list.append(wav_path)
	if hps.prep and hps.pth is not None:
		with open(hps.pth, 'wb') as w:
			pickle.dump(f_list, w)
	return f_list


class ljdataset(Dataset):
	def __init__(self, fdir):
		if hps.prep and hps.pth is not None and os.path.isfile(hps.pth):
			with open(hps.pth, 'rb') as r:
				self.f_list = pickle.load(r)
		else:
			self.f_list = files_to_list(fdir)

	def __getitem__(self, index):
		if hps.prep:
			wav, mel = self.f_list[index]
			seg_ml = hps.seg_l//hps.frame_shift+1
			ms = np.random.randint(0, mel.shape[1]-seg_ml) if mel.shape[1] > seg_ml else 0
			ws = hps.frame_shift*ms
			wav = wav[ws:ws+hps.seg_l]
			mel = mel[:, ms:ms+seg_ml]
		else:
			wav = load_wav(self.f_list[index])
			mel = melspectrogram(wav).astype(np.float32)
		return wav, mel

	def __len__(self):
		return len(self.f_list)


def collate_fn(batch):
	wavs = []
	mels = []
	for wav, mel in batch:
		wavs.append(wav)
		mels.append(mel)
	wavs = torch.Tensor(wavs)
	mels = torch.Tensor(mels)
	return wavs, mels
