import torch.nn as nn

class TransformerEmbedding(nn.Module):
	def __init__(self, embedding, token_embed, pos_embed, dr_rate=0):
		super(TransformerEmbedding, self).__init__
		self.embedding = nn.Sequential(token_embed, pos_embed)
		self.dropout = nn.Dropout(p=dr_rate)