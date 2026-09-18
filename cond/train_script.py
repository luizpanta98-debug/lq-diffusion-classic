from unet import UNet
from data import load_imagefolder, get_subset_targets
from dm import DM
from seed import set_seed
from tqdm import tqdm
import numpy as np
import json
import os 
import torch as th
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

def train(net, dm, train_loader, val_loader, config):
	with open(f"{config['save_checkpoint']}/config.json", "w") as f:
		json.dump(config, f, indent=4)

	if config['ema']:
		ema = AveragedModel(net,
		multi_avg_fn=get_ema_multi_avg_fn(config['ema_decay']))
		if config['load_checkpoint']:
			ema.load_state_dict(th.load(f"{config['save_checkpoint']}/weights/ema.pt", weights_only=False))

	opt = th.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))
	if config['load_checkpoint']:
		try:
			opt.load_state_dict(th.load(f"{config['save_checkpoint']}/weights/opt.pt"))
			print('opt loaded')
		except:
			print('couldnt load opt! training with new opt')

	best_val_path = f"{config['save_checkpoint']}/weights/best_val.txt"
	best_val = float('inf')
	if config['load_checkpoint']:
		try:
			with open(best_val_path, "r") as f:
				best_val = float(f.read().strip())
			print(f'best_val loaded: {best_val}')
		except:
			print('couldnt load best_val! starting from inf')

	acc_steps = config['batch_size'] // config['virtual_batch_size']
	for epoch in range(1,config['epochs']+1):
		# EPOCH IN
		loss_epoch=[]
		val_epoch=[]
		val_per_class = {c: [] for c in range(config['label_dim'])}
		for i, (images,labels) in enumerate(tqdm(train_loader)):
			#opt.zero_grad()
			images = images.to(config['device'])
			# labels=None
			labels_int = labels.to(config['device'])
			labels = F.one_hot(labels_int, num_classes=config['label_dim'])
			ts = th.randint(0, dm.T, (images.shape[0], ), device=config['device'])
			xts, eps = dm.q(images, ts)
			eps_pred = net(xts, ts, labels)
			loss = F.mse_loss(eps_pred, eps)
			loss_epoch.append(loss.item())
			(loss / acc_steps).backward()
			#opt.step()
			if (i+1)%acc_steps == 0:
				th.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
				opt.step()
				opt.zero_grad()
				if config['ema']:
					ema.update_parameters(net)

		with th.inference_mode():
			for i, (images, labels) in enumerate(tqdm(val_loader)):
				images = images.to(config['device'])
				# labels=None
				labels_int = labels.to(config['device'])
				labels = F.one_hot(labels_int, num_classes=config['label_dim'])
				ts = th.randint(0, dm.T, (images.shape[0], ), device=config['device'])
				xts, eps = dm.q(images, ts)
				eps_pred = net(xts, ts, labels)
				val_loss = F.mse_loss(eps_pred, eps)
				per_sample = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[1,2,3])
				val_epoch.append(val_loss.item())
				for c in labels_int.unique():
					mask = (labels_int == c)
					val_per_class[c.item()].append(per_sample[mask].mean().item())

		mean_val = np.mean(val_epoch)

		if epoch % config['save_freq'] == 0:
			th.save(net.state_dict(), f"{config['save_checkpoint']}/weights/last_weights.pt")
			print(f"model weights saved -> {config['save_checkpoint']}/weights/last_weights.pt")
			th.save(opt.state_dict(), f"{config['save_checkpoint']}/weights/opt.pt")
			print(f"opt weights saved -> {config['save_checkpoint']}/weights/opt.pt")
			if config['ema']:
				th.save(ema.state_dict(), f"{config['save_checkpoint']}/weights/ema.pt")
				print(f"ema saved -> {config['save_checkpoint']}/weights/ema.pt")
				th.save(ema.module.state_dict(), f"{config['save_checkpoint']}/weights/ema_weights.pt")
				print(f"ema weights saved -> {config['save_checkpoint']}/weights/ema_weights.pt")

		if mean_val < best_val:
			best_val = mean_val
			with open(best_val_path, "w") as f:
				f.write(str(best_val))
			th.save(net.state_dict(), f"{config['save_checkpoint']}/weights/best_weights.pt")
			print(f"best model updated -> {config['save_checkpoint']}/weights/best_weights.pt (val={round(best_val,4)})")

		print(f'\nepoch {epoch}\nloss {round(np.mean(loss_epoch),4)}\nval {round(mean_val,4)}\n')
		for c, vals in val_per_class.items():
			if vals:
				print(f'  class {c}: val {round(np.mean(vals),4)} (n={len(vals)})')
		print()
		# EPOCH OUT

if __name__ == "__main__":
	# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	with open("config.json", "r") as f:
		config = json.load(f)

	set_seed(config['seed'])
	os.makedirs(f"{config['save_checkpoint']}/weights", exist_ok=True)

	train_dataset, val_dataset = load_imagefolder(config['data_fp'], config['img_resolution'], config['split'])

	# WeightedRandomSampler
	targets = np.array(get_subset_targets(train_dataset))
	class_counts = np.bincount(targets, minlength=config['label_dim'])
	class_weights = 1.0 / np.clip(class_counts, 1, None)
	sample_weights = class_weights[targets]
	gen = th.Generator().manual_seed(config['seed'])
	train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=gen)

	# num_workers>0 
	num_workers = min(8, os.cpu_count() or 1)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config['virtual_batch_size'],
		sampler=train_sampler,
		drop_last=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=num_workers > 0,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config['virtual_batch_size'],
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=num_workers > 0,
	)
	dm = DM()
	net = UNet(
		img_resolution=config["img_resolution"],
		in_channels=config["in_channels"], 
		out_channels=config["out_channels"], 
		label_dim=config["label_dim"],
		model_channels=config["model_channels"], 
		channel_mult=config["channel_mult"],
		channel_mult_emb=config["channel_mult_emb"],
		num_res_blocks=config["num_res_blocks"],
		attn_resolutions=config["attn_resolutions"],
		).to(config['device'])

	if config['load_checkpoint']:	
		net.load_state_dict(th.load(
				f"{config['save_checkpoint']}/weights/last_weights.pt",
				weights_only=False))
		print('last weights loaded.')

	print(f'train data size: {len(train_dataset)}\nval data size: {len(val_dataset)}')
	train(net, dm, train_loader, val_loader, config)