# WG-WaveNet: Real-Time High-Fidelity Speech Synthesis without GPU

### Po-chun Hsu, Hung-yi Lee

In our recent [paper](https://arxiv.org/abs/2005.07412), we propose WG-WaveNet, a fast, lightweight, and high-quality waveform generation model. WG-WaveNet is composed of a compact flow-based model and a post-filter. The two components are jointly trained by maximizing the likelihood of the training data and optimizing loss functions on the frequency domains. As we design a flow-based model that is heavily compressed, the proposed model requires much less computational resources compared to other waveform generation models during both training and inference time; even though the model is highly compressed, the post-filter maintains the quality of generated waveform. Our PyTorch implementation can be trained using less than 8 GB GPU memory and generates audio samples at a rate of more than 5000 kHz on an NVIDIA 1080Ti GPU. Furthermore, even if synthesizing on a CPU, we show that the proposed method is capable of generating 44.1 kHz speech waveform 1.2 times faster than real-time. Experiments also show that the quality of generated audio is comparable to those of other methods.

Visit the [demopage](https://bogihsu.github.io/WG-WaveNet/) for audio samples.

## TODO
- [ ] Release pretrained model.
- [ ] Combine with [Tacotron2](https://github.com/BogiHsu/Tacotron2-PyTorch).

## Requirements
- Python >= 3.5.2
- torch >= 1.4.0
- numpy
- scipy
- pickle
- librosa
- tensorboardX

## Preprocessing


## Training
1. Download [LJ Speech](https://keithito.com/LJ-Speech-Dataset/). In this example it's in `data/`

2. For training Tacotron2, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models>
```

3. For training using a pretrained model, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models> --ckpt_pth=<pth/to/pretrained/model>
```

4. For using Tensorboard (optional), run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models> --log_dir=<dir/to/logs>
```

## Inference
- For synthesizing wav files, run the following command.

```bash
python3 inference.py --ckpt_pth=<pth/to/model> --src_pth=<pth/to/src/wavs> --res_pth=<pth/to/save/wavs>
```

## Pretrained Model
Work in progress.

## TTS
We will combine this vocoder with Tacotron2. More information and Colab demo will be released [here](https://github.com/BogiHsu/Tacotron2-PyTorch). 

## References
- [WaveGlow by NVIDIA](https://github.com/NVIDIA/waveglow)
- [ParallelWaveGAN by kan-bayashi](https://github.com/kan-bayashi/ParallelWaveGAN)
