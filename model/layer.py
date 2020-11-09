import torch
from torch.autograd import Variable


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
	n_channels_int = n_channels[0]
	in_act = input_a+input_b
	t_act = torch.tanh(in_act[:, :n_channels_int, :])
	s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
	acts = t_act * s_act
	return acts


class Invertible1x1Conv(torch.nn.Module):
	"""
	The layer outputs both the convolution, and the log determinant
	of its weight matrix.  If reverse=True it does convolution with
	inverse
	"""
	def __init__(self, c):
		super(Invertible1x1Conv, self).__init__()
		self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
									bias=False)

		# Sample a random orthonormal matrix to initialize weights
		W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

		# Ensure determinant is 1.0 not -1.0
		if torch.det(W) < 0:
			W[:,0] = -1*W[:,0]
		W = W.view(c, c, 1)
		self.conv.weight.data = W

	def forward(self, z, reverse = False):
		# shape
		batch_size, group_size, n_of_groups = z.size()

		W = self.conv.weight.squeeze()

		if reverse:
			if not hasattr(self, 'set'):
				# Reverse computation
				W_inverse = W.float().inverse()
				W_inverse = W_inverse[..., None]
				self.W_inverse = W_inverse
			z = torch.nn.functional.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
			return z
		else:
			# Forward computation
			log_det_W = batch_size * n_of_groups * torch.logdet(W)
			z = self.conv(z)
			return z, log_det_W
	def set_inverse(self):
		W = self.conv.weight.squeeze()
		W_inverse = W.float().inverse()
		W_inverse = W_inverse[..., None]
		self.W_inverse = W_inverse
		self.set = True

