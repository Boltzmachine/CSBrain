import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from tqdm import tqdm
from PIL import Image
import h5py
import os
from pathlib import Path


IMAGE_EXTENSIONS = {
	".jpg",
	".jpeg",
	".png",
	".bmp",
	".webp",
	".tiff",
	".tif",
}

def find_images(root):
	image_paths = []
	for path in root.rglob("*"):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			image_paths.append(path)
	image_paths.sort()
	return image_paths

def main():
	model_name = "facebook/dinov2-base"
	output_dir = Path("data/THINGS-EEG2")
	output_dir.mkdir(parents=True, exist_ok=True)
	storage_dtype = np.float32
	h5f = h5py.File(output_dir / "pixel_values.h5", "w")

	for split in ['train', 'test']:
		if split == 'train':
			image_paths = find_images(Path("data/THINGS-EEG2/training_images"))
		else:
			image_paths = find_images(Path("data/THINGS-EEG2/test_images"))
		if not image_paths:
			raise ValueError(f"No images found under: {Path('data/THINGS-EEG2/training_images')} or {Path('data/THINGS-EEG2/test_images')}")

		processor = AutoImageProcessor.from_pretrained(model_name)

		written = 0
		all_image_arr = None #np.zeros((len(image_paths), 3, processor.size[], processor.size["width"]), dtype=storage_dtype)
		written = set()
		
		image_paths_with_indices = []
		for image_path in image_paths:
			class_idx = int(image_path.parent.name.split("_")[0])
			pic_idx = int(image_path.stem.rsplit("_", 1)[1][:-1])
			image_paths_with_indices.append((image_path, class_idx, pic_idx))
		image_paths_with_indices.sort(key=lambda x: (x[1], x[2]))
		image_paths = [x[0] for x in image_paths_with_indices]

		grp = h5f.create_group(split)

		for image_index, image_path in tqdm(
			enumerate(image_paths),
			total=len(image_paths)
		):
			image = Image.open(image_path).convert("RGB")
			inputs = processor(images=image, return_tensors="pt")
			pixel_values = inputs["pixel_values"][0].cpu().numpy().astype(storage_dtype, copy=False)
			if all_image_arr is None:
				all_image_arr = np.zeros((len(image_paths), *pixel_values.shape), dtype=storage_dtype)
			all_image_arr[image_index] = pixel_values
			if image_index not in written:
				written.add(image_index)
			else:
				raise ValueError(f"Duplicate image index detected: {image_index} from path: {image_path}")
			
		grp.create_dataset("pixel_values", data=all_image_arr)
	assert len(written) == len(image_paths), f"Expected to write {len(image_paths)} unique images, but got {len(written)} unique indices."

	h5f.close()
	print(f"Saved {written} processed images to")
	

if __name__ == "__main__":
	main()