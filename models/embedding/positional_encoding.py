import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

	def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
		super(PositionalEncoding, self).__init__()
		encoding = torch.zeros(max_len, d_embed)
		encoding.requires_grad = False
		position = torch.arange(0, max_len).float().unsqueeze(1)