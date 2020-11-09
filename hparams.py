class hparams:
	seed = 0
	################################
	# Audio                        #
	################################
	num_mels = 80
	num_freq = 513
	sample_rate = 22050
	frame_shift = 200
	frame_length = 800
	preemphasis = 0.97
	min_level_db = -100
	ref_level_db = 20
	fmin = 0
	fmax = 8000
	seg_l = 16000

	################################
	# Train	                       #
	################################
	is_cuda = True
	pin_mem = True
	n_workers = 4
	prep = False
	pth = None
	lr = 4e-4
	sch = True
	sch_step = 200e3
	sch_g = 0.5
	sch_stop = 800e3
	max_iter = 1000e3
	batch_size = 8
	gn = 10
	n = 3
	iters_per_log = n*(10//n)
	iters_per_sample = n*(500//n)
	iters_per_ckpt = 10000

	################################
	# Model                        #
	################################
	up_scale = [2, 5, 2, 5, 2] # assert product = frame_shift
	sigma = 0.6
	n_flows = 4
	n_group = 8
	# for WN
	n_layers = 7
	n_channels = 128
	kernel_size = 3
	# for PF
	PF_n_layers = 7
	PF_n_channels = 64

	################################
	# Spectral Loss                #
	################################
	mag = True
	mel = True
	fft_sizes = [2048, 1024, 512, 256, 128]
	hop_sizes = [400, 200, 100, 50, 25]
	win_lengths = [2000, 1000, 500, 200, 100]
	mel_scales = [4, 2, 1, 0.5, 0.25]
