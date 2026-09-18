import os
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from unet import UNet
from dm import DM
from data import load_imagefolder


def ddim_cond_sample(net, dm, config, batch_size=10, num_samples=100, out_dir='gen_samples', ds=25, eta=1.0):
	os.makedirs(out_dir, exist_ok=True)
	
	subseq = [dm.T-1]
	subseq = subseq + list(reversed(range(ds, dm.T, ds)))
	train_data, val_data = load_imagefolder(config['data_fp'], config['img_resolution'], config['split'])
	class_dict = train_data.dataset.class_to_idx
	class_dict = {idx:classname for classname,idx in class_dict.items()}
	lbls = [idx for idx,_ in class_dict.items()]

	for lbl in lbls:
		img_num = 0
		with th.inference_mode():
			for _ in range(0, num_samples, batch_size):
				xt = th.randn(
					batch_size,
					config['in_channels'], 
					config['img_resolution'], 
					config['img_resolution'],
					device=config['device'])

				for idx,time in tqdm(list(enumerate(subseq))):
					ts = th.full((batch_size,), time, dtype=th.long, device=config['device'])
					if idx < len(subseq) - 1:
						tsprev = th.full((batch_size,), subseq[idx+1], dtype=th.long, device=config['device'])
					else:
						tsprev = th.full((batch_size,), 0, dtype=th.long, device=config['device'])

					y = F.one_hot(th.tensor(lbl, device=config['device']), num_classes=config['label_dim']).float()

					xt, x0, eps = dm.ddim_sample(
										eta, 
										net, 
										xt, 
										ts, 
										tsprev, 
										y)
				xt = xt.detach().cpu().permute(0,2,3,1).numpy()
				for np_img in xt:
					np_img = (np_img + 1)/2 #[-1,1]
					np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
					img = Image.fromarray(np_img)
					img.save(f"{out_dir}/{class_dict[lbl]}_{img_num}.png")
					img_num+=1


def cond_sample(net, dm, config, batch_size=10, num_samples=20, out_dir='gen_samples'):
	os.makedirs(out_dir, exist_ok=True)

	train_data, val_data = load_imagefolder(config['data_fp'], config['img_resolution'], config['split'])
	class_dict = train_data.dataset.class_to_idx
	class_dict = {idx:classname for classname,idx in class_dict.items()}
	lbls = [idx for idx,_ in class_dict.items()]
	img_num = 0

	for y in lbls:
		with th.inference_mode():
			for _ in range(0, num_samples, batch_size):
				xt = th.randn(
					batch_size,
					config['in_channels'], 
					config['img_resolution'], 
					config['img_resolution'],
					device=config['device'])

				for ts in tqdm( list(reversed(range(dm.T))) ):
					ts = th.full((batch_size,), ts, dtype=th.long, device=config['device'])
					ys = th.full((batch_size,), y, dtype=th.long, device=config['device'])
					ys = F.one_hot(ys, num_classes=config['label_dim'])
					xt = dm.p(xt, ts, ys, net)

				xt = xt.detach().cpu().permute(0,2,3,1).numpy()
				for np_img in xt:
					np_img = (np_img + 1)/2 #[-1,1]
					np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
					img = Image.fromarray(np_img)
					img.save(f"{out_dir}/{class_dict[y]}{img_num}.png")
					img_num+=1


if __name__ == "__main__":
	with open("config.json", "r") as f:
		config = json.load(f)

	set_seed(config['seed'])

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


	net.load_state_dict(th.load(
			f"{config['save_checkpoint']}/weights/ema_weights.pt",
			weights_only=False))
	print('weights loaded.')

	#cond_sample(net, dm, config)
	ddim_cond_sample(net, dm, config)

