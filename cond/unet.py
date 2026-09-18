'''
This unet.py is adapted from github.com/NVlabs/edm
'''
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def silu(x):
	return F.silu(x)

def gnorm32(channels):
	return nn.GroupNorm(32, channels)

# Attention from EDM repo.
# Source: github.com/NVlabs/edm/blob/main/training/networks.py

class AttentionOp(th.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = th.einsum('ncq,nck->nqk', q.to(th.float32), (k / np.sqrt(k.shape[1])).to(th.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = th._softmax_backward_data(grad_output=dw.to(th.float32), output=w.to(th.float32), dim=2, input_dtype=th.float32)
        dq = th.einsum('nck,nqk->ncq', k.to(th.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = th.einsum('ncq,nqk->nck', q.to(th.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


# PositionalEmbedding 
class PositionalEmbedding(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, t):
		device = t.device
		half_dim = self.dim // 2
		embeddings = math.log(10000) / (half_dim - 1)
		embeddings = th.exp(th.arange(half_dim, device=device) * -embeddings)
		embeddings = t[:, None] * embeddings[None, :]
		embeddings = th.cat((embeddings.sin(), embeddings.cos()), dim=-1)
		return embeddings # (B, dim)


class Conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, up=False, down=False):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels 
		self.up = up
		self.down = down 
		if self.up:
			self.main = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
		elif self.down:
			self.main = nn.Conv2d(in_channels, out_channels, 2, stride=2)
		else:
			self.main = nn.Conv2d(in_channels, out_channels, 3, padding=1)
	def forward(self,x):
		return self.main(x)


class UNetBlock(nn.Module):
	def __init__(self, in_channels, out_channels, emb_channels, num_heads=1, up=False, down=False, attn=False):
		super().__init__()
		self.num_heads=num_heads
		self.in_channels=in_channels
		self.out_channels=out_channels
		self.up = up 
		self.attn = attn
		self.norm0 = gnorm32(in_channels)
		self.norm1 = gnorm32(out_channels)
		self.affine = nn.Linear(emb_channels, out_channels)
		self.skip = Conv2d(in_channels, out_channels, up=up, down=down)
		self.conv0 = Conv2d(in_channels, out_channels, up=up, down=down)
		self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

		if self.attn:
			self.norm2 = gnorm32(out_channels)
			self.qkv = nn.Conv2d(out_channels, 3*out_channels, 1)
			self.proj = nn.Conv2d(out_channels, out_channels, 1)

	def forward(self, x, emb):
		orig = x 
		x = self.conv0(silu(self.norm0(x)))
		params = self.affine(emb).unsqueeze(2).unsqueeze(3)
		x = silu(self.norm1(x + params))
		x = self.conv1(x)
		x = x + self.skip(orig)

		if self.attn:
			q,k,v = self.qkv(self.norm2(x)).reshape(x.shape[0]*self.num_heads, x.shape[1]//self.num_heads, 3, -1).unbind(2)
			w = AttentionOp.apply(q, k)
			a = th.einsum('nqk,nck->ncq', w, v)
			x = x + self.proj(a.reshape(*x.shape))
		return x

class UNet(nn.Module):
	def __init__(self, 
		img_resolution,
		in_channels, 
		out_channels, 
		label_dim, # = 0 Uncond
		model_channels=64, 
		channel_mult=[1,2,2,2],
		channel_mult_emb=4, # Multiplier for the dimensionality of the embedding vector.
		num_res_blocks=4, # Number of residual blocks per resolution.
		attn_resolutions=[16],
		):
		super().__init__()

		emb_channels = model_channels * channel_mult_emb
		noise_channels = model_channels * 1 # DDPM

		# Mapping t and y
		self.map_noise = PositionalEmbedding(noise_channels)
		self.map_label = nn.Linear(label_dim, noise_channels) if label_dim else None
		self.map_layer0 = nn.Linear(noise_channels, emb_channels)
		self.map_layer1 = nn.Linear(emb_channels, emb_channels)

		# Encoder
		self.enc = nn.ModuleDict()
		cout = in_channels
		caux = in_channels
		for level,mult in enumerate(channel_mult):
			res = img_resolution >> level
			if level == 0:
				cin = cout 
				cout = model_channels 
				self.enc[f'{res}x{res}_conv'] = nn.Conv2d(cin, cout, 3, padding=1)
			else:
				self.enc[f'{res}x{res}_down'] = UNetBlock(cout, cout, emb_channels, down=True)

			for idx in range(num_res_blocks):
				cin = cout
				cout = model_channels * mult 
				attn = (res in attn_resolutions)
				self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, emb_channels, attn=attn)
		skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

		# Decoder
		self.dec = nn.ModuleDict()
		for level, mult in reversed(list(enumerate(channel_mult))):
			res = img_resolution >> level
			if level == len(channel_mult) - 1:
				self.dec[f'{res}x{res}_in0'] = UNetBlock(cout, cout, emb_channels, attn=True)
				self.dec[f'{res}x{res}_in1'] = UNetBlock(cout, cout, emb_channels)
			else:
				self.dec[f'{res}x{res}_up'] = UNetBlock(cout, cout, emb_channels, up=True)
			for idx in range(num_res_blocks+1):
				cin = cout + skips.pop()
				cout = model_channels * mult 
				attn = (idx==num_res_blocks and res in attn_resolutions)
				self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, emb_channels, attn=attn)
			if level == 0:
				self.dec[f'{res}x{res}_aux_norm'] = gnorm32(cout)
				self.dec[f'{res}x{res}_aux_conv'] = nn.Conv2d(cout, out_channels, 3, padding=1)


	def forward(self, x, noise_labels, class_labels):
		# Mapping
		emb = self.map_noise(noise_labels)
		# emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) #swap sin/cos
		if self.map_label is not None:
			tmp = class_labels 
			emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))

		emb = silu(self.map_layer0(emb))
		emb = silu(self.map_layer1(emb))

		# Encoder
		skips = []
		aux = x # We removed auxiliary blocks in Encoder.
		for name, block in self.enc.items():
			x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
			skips.append(x)

		# Decoder
		aux=None
		tmp=None
		for name, block in self.dec.items():
			if 'aux_norm' in name:
				tmp = block(x)
			elif 'aux_conv' in name:
				tmp = block(silu(tmp))
				aux = tmp if aux is None else tmp+aux
			else:
				if x.shape[1] != block.in_channels:
					x = th.cat([x, skips.pop()], dim=1)
				x = block(x, emb)
		return aux









