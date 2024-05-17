import math
import torch.nn as nn

class TokenEmbedding(nn.Module):
	def __init__(self, d_embed, vocab_size):
		# inherit 할 때, class 이름(자기자신)과 self를 넣어주는 걸로 상속 대상을 구체화시켜줌
		super(TokenEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, d_embed)
		self.d_embed = d_embed
	
	def forward(self, x):
		out = self.embedding(x) * math.sqrt(self.d_embed)
		return out

embedding = TokenEmbedding(20, 4)
print(embedding)