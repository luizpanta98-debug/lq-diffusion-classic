from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# util for weighted sampler
def get_subset_targets(subset):
	return [subset.dataset.targets[i] for i in subset.indices]

def is_valid_file(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        print(f"Skipping damaged file: {path}")
        return False

def load_imagefolder(fp, img_resolution, split):
	
	transform = transforms.Compose([
		transforms.Resize((img_resolution, img_resolution)),
		transforms.ToTensor(), #[0,1]
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #[-1,1]
		])

	dataset = ImageFolder(root=fp, transform=transform, is_valid_file=is_valid_file)
	train_size = int(split * len(dataset))
	val_size = len(dataset) - train_size

	train_dataset, val_dataset = random_split(dataset,[train_size, val_size])	
	return train_dataset, val_dataset


def load_imagedataset(fp, img_resolution, split):

	transform = transforms.Compose([
		transforms.Resize((img_resolution, img_resolution)),
		transforms.ToTensor(), #[0,1]
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #[-1,1]

		])
	dataset = ImageDataset(fp, transform=transform)
	train_size = int(split * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset,[train_size, val_size])	
	return train_dataset, val_dataset