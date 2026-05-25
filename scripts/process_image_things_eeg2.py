import argparse
import numpy as np
from transformers import AutoImageProcessor, AutoVideoProcessor
from tqdm import tqdm
from PIL import Image
import h5py
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))
from datasets.cached_dataset import pixel_values_filename, _encoder_kind


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


def _load_vision_processor(model_name: str):
	if _encoder_kind(model_name) == 'vjepa2':
		return AutoVideoProcessor.from_pretrained(model_name)
	return AutoImageProcessor.from_pretrained(model_name)


def _processed_pixel_values(processor, model_name: str, image) -> np.ndarray:
	if _encoder_kind(model_name) == 'vjepa2':
		frame = np.array(image)
		out = processor(videos=[[frame]], return_tensors='pt')
		return out['pixel_values_videos'][0, 0].cpu().numpy()
	out = processor(images=image, return_tensors='pt')
	return out['pixel_values'][0].cpu().numpy()


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Build THINGS-EEG2 pixel_values H5 for a given vision encoder.')
	p.add_argument('--model-name', type=str, default='facebook/dinov2-base',
		help='HF model id whose processor we run (e.g. facebook/dinov2-base, '
		     'facebook/vjepa2-vitl-fpc64-256). Output filename pairs with the encoder.')
	p.add_argument('--output-dir', type=Path, default=Path('data/THINGS-EEG2'),
		help='Where to write the H5. Filename is chosen via pixel_values_filename.')
	return p.parse_args()


def main():
	args = parse_args()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	storage_dtype = np.float32
	out_path = output_dir / pixel_values_filename(args.model_name)
	h5f = h5py.File(out_path, "w")

	processor = _load_vision_processor(args.model_name)

	total_written = 0
	for split in ['train', 'test']:
		if split == 'train':
			image_paths = find_images(Path("data/THINGS-EEG2/training_images"))
		else:
			image_paths = find_images(Path("data/THINGS-EEG2/test_images"))
		if not image_paths:
			raise ValueError(f"No images found under: {Path('data/THINGS-EEG2/training_images')} or {Path('data/THINGS-EEG2/test_images')}")

		all_image_arr = None
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
			total=len(image_paths),
			desc=f'{split}',
		):
			image = Image.open(image_path).convert("RGB")
			pixel_values = _processed_pixel_values(
				processor, args.model_name, image
			).astype(storage_dtype, copy=False)
			if all_image_arr is None:
				all_image_arr = np.zeros((len(image_paths), *pixel_values.shape), dtype=storage_dtype)
			all_image_arr[image_index] = pixel_values
			if image_index not in written:
				written.add(image_index)
			else:
				raise ValueError(f"Duplicate image index detected: {image_index} from path: {image_path}")

		grp.create_dataset("pixel_values", data=all_image_arr)
		assert len(written) == len(image_paths), (
			f"Expected to write {len(image_paths)} unique images for split={split}, "
			f"but got {len(written)} unique indices."
		)
		total_written += len(written)

	h5f.close()
	print(f"Saved {total_written} processed images to {out_path}")


if __name__ == "__main__":
	main()
