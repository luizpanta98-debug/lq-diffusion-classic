'''FID and Inception Score across DDIM step sizes (ds). Only the two plots are written to disk.'''
# pip install pytorch-fid scienceplots

import os
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt

from unet import UNet
from dm import DM
from seed import set_seed
from data import is_valid_file

DS_VALUES = [25, 50, 100]
IS_SPLITS = 10
GEN_BATCH = 10   # diffusion sampling batch, raise if you have GPU headroom
FEAT_BATCH = 50  # inception forward batch for real images


def load_config():
	with open('config.json', 'r') as f:
		return json.load(f)


def build_net(config):
	net = UNet(
		img_resolution=config['img_resolution'],
		in_channels=config['in_channels'],
		out_channels=config['out_channels'],
		label_dim=config['label_dim'],
		model_channels=config['model_channels'],
		channel_mult=config['channel_mult'],
		channel_mult_emb=config['channel_mult_emb'],
		num_res_blocks=config['num_res_blocks'],
		attn_resolutions=config['attn_resolutions'],
	).to(config['device'])
	net.load_state_dict(th.load(
		f"{config['save_checkpoint']}/weights/ema_weights.pt",
		weights_only=False))
	net.eval()
	print('weights loaded.')
	return net


def real_dataset_and_counts(config):
	# [0,1] range, no normalize -> straight into the inception nets
	transform = transforms.Compose([
		transforms.Resize((config['img_resolution'], config['img_resolution'])),
		transforms.ToTensor(),
	])
	# same is_valid_file filter as load_imagefolder, so counts match what training actually saw
	dataset = ImageFolder(root=config['data_fp'], transform=transform, is_valid_file=is_valid_file)
	targets = np.array(dataset.targets)
	counts = {c: int((targets == c).sum()) for c in dataset.class_to_idx.values()}
	return dataset, counts


def get_fid_model(device):
	block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
	model = InceptionV3([block_idx]).to(device)
	model.eval()
	return model


def get_is_model(device):
	model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
	model.to(device)
	model.eval()
	return model


@th.inference_mode()
def fid_feats(imgs, model):
	# imgs in [0,1], NCHW
	pred = model(imgs)[0]
	if pred.size(2) != 1 or pred.size(3) != 1:
		pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
	return pred.squeeze(3).squeeze(2).cpu().numpy()


@th.inference_mode()
def is_probs(imgs, model):
	x = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
	mean = th.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
	std = th.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
	x = (x - mean) / std
	logits = model(x)
	return F.softmax(logits, dim=1).cpu().numpy()


@th.inference_mode()
def real_stats(dataset, device, fid_model):
	loader = th.utils.data.DataLoader(
		dataset, batch_size=FEAT_BATCH, shuffle=False,
		num_workers=min(8, os.cpu_count() or 1))
	feats = []
	for imgs, _ in loader:
		feats.append(fid_feats(imgs.to(device), fid_model))
	feats = np.concatenate(feats, axis=0)
	return feats.mean(axis=0), np.cov(feats, rowvar=False)


def ddim_schedule(T, ds):
	return [T - 1] + list(reversed(range(ds, T, ds)))


@th.inference_mode()
def sample_class_feats(net, dm, config, subseq, label_idx, n_samples, device, fid_model, is_model):
	y = F.one_hot(th.tensor(label_idx, device=device), num_classes=config['label_dim']).float()
	feats, probs = [], []
	done = 0
	while done < n_samples:
		bs = min(GEN_BATCH, n_samples - done)
		xt = th.randn(bs, config['in_channels'], config['img_resolution'], config['img_resolution'], device=device)
		for idx, t in enumerate(subseq):
			ts = th.full((bs,), t, dtype=th.long, device=device)
			tprev = th.full((bs,), subseq[idx + 1] if idx < len(subseq) - 1 else 0, dtype=th.long, device=device)
			xt, _, _ = dm.ddim_sample(1.0, net, xt, ts, tprev, y)
		imgs = ((xt + 1) / 2).clamp(0, 1)
		feats.append(fid_feats(imgs, fid_model))
		probs.append(is_probs(imgs, is_model))
		done += bs
	return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0)


def inception_score(probs, splits=IS_SPLITS):
	n = probs.shape[0]
	scores = []
	for i in range(splits):
		part = probs[i * n // splits: (i + 1) * n // splits]
		py = part.mean(axis=0, keepdims=True)
		kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(axis=1)
		scores.append(np.exp(kl.mean()))
	return float(np.mean(scores)), float(np.std(scores))


def set_style():
	try:
		import scienceplots
		plt.style.use(['science', 'no-latex'])
	except ImportError:
		plt.rcParams.update({
			'font.family': 'serif',
			'font.size': 11,
			'axes.grid': True,
			'grid.alpha': 0.3,
			'axes.spines.top': False,
			'axes.spines.right': False,
		})
	plt.rcParams['figure.dpi'] = 150


def plot_fid(ds_values, fid_values, out_path):
	fig, ax = plt.subplots(figsize=(4.5, 3.5))
	ax.plot(ds_values, fid_values, marker='o', linewidth=1.8)
	ax.set_xlabel('DDIM step size (ds)')
	ax.set_ylabel('FID')
	ax.set_xticks(ds_values)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_is(ds_values, is_means, is_stds, out_path):
	fig, ax = plt.subplots(figsize=(4.5, 3.5))
	ax.errorbar(ds_values, is_means, yerr=is_stds, marker='o', linewidth=1.8, capsize=3)
	ax.set_xlabel('DDIM step size (ds)')
	ax.set_ylabel('Inception Score')
	ax.set_xticks(ds_values)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def main():
	config = load_config()
	device = config['device']
	set_seed(config['seed'])
	th.backends.cudnn.benchmark = True
	os.makedirs('plots', exist_ok=True)

	dm = DM()
	net = build_net(config)

	dataset, counts = real_dataset_and_counts(config)
	print(f'real dataset: {len(dataset)} images, {len(counts)} classes, counts={counts}')

	fid_model = get_fid_model(device)
	is_model = get_is_model(device)

	mu_real, sigma_real = real_stats(dataset, device, fid_model)
	print('real stats computed.')

	fid_values, is_means, is_stds = [], [], []
	for ds in DS_VALUES:
		subseq = ddim_schedule(dm.T, ds)
		print(f'ds={ds} ({len(subseq)} steps): sampling...')

		all_feats, all_probs = [], []
		for label_idx, n in counts.items():
			feats, probs = sample_class_feats(net, dm, config, subseq, label_idx, n, device, fid_model, is_model)
			all_feats.append(feats)
			all_probs.append(probs)
		all_feats = np.concatenate(all_feats, axis=0)
		all_probs = np.concatenate(all_probs, axis=0)

		mu_fake = all_feats.mean(axis=0)
		sigma_fake = np.cov(all_feats, rowvar=False)
		fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
		is_mean, is_std = inception_score(all_probs)

		print(f'ds={ds}: FID={fid:.3f}  IS={is_mean:.3f}+-{is_std:.3f}')
		fid_values.append(fid)
		is_means.append(is_mean)
		is_stds.append(is_std)

	set_style()
	plot_fid(DS_VALUES, fid_values, 'plots/fid_vs_ds.png')
	plot_is(DS_VALUES, is_means, is_stds, 'plots/inception_score_vs_ds.png')
	print('plots saved -> ./plots')


if __name__ == '__main__':
	main()