import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model
import h5py
import numpy as np


IMAGE_EXTENSIONS = {
	".jpg",
	".jpeg",
	".png",
	".bmp",
	".webp",
	".tiff",
	".tif",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract per-layer DINOv2 embeddings for all images in a directory.",
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
        default="data/Alljoined-1.6M/images",
		help="Root directory to recursively search for images.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
        default="data/Alljoined-1.6M/",
		help="Directory where per-image embedding files will be saved.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default="facebook/dinov2-base",
		help="Hugging Face DINOv2 model identifier.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to run inference on (e.g., cuda, cuda:0, cpu).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		help="Batch size for inference.",
	)
	parser.add_argument(
		"--save-cls-only",
		action="store_true",
		help="If set, save only CLS embedding per layer instead of all patch tokens.",
	)
	parser.add_argument(
		"--dtype",
		type=str,
		default="float32",
		choices=["float32", "float16"],
		help="Storage dtype for saved embeddings.",
	)
	return parser.parse_args()


def find_images(root: Path) -> List[Path]:
	image_paths: List[Path] = []
	for path in root.rglob("*"):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			image_paths.append(path)
	image_paths.sort()
	return image_paths


def to_chunks(items: List[Path], size: int) -> Iterable[List[Path]]:
	for i in range(0, len(items), size):
		yield items[i : i + size]


def load_images(paths: List[Path]) -> List[Image.Image]:
	images: List[Image.Image] = []
	for path in paths:
		with Image.open(path) as img:
			images.append(img.convert("RGB"))
	return images


def tensor_cast_dtype(dtype_name: str) -> torch.dtype:
	if dtype_name == "float16":
		return torch.float16
	return torch.float32


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
	rel = input_path.relative_to(input_root)
	return output_root / rel.with_suffix(".pt")


def main() -> None:
	args = parse_args()

	if not args.input_dir.exists() or not args.input_dir.is_dir():
		raise ValueError(f"Input directory does not exist or is not a directory: {args.input_dir}")
	if args.batch_size <= 0:
		raise ValueError("batch-size must be a positive integer")

	args.output_dir.mkdir(parents=True, exist_ok=True)

	image_paths = find_images(args.input_dir)
	if not image_paths:
		print(f"No images found under: {args.input_dir}")
		return

	print(f"Found {len(image_paths)} images.")
	print(f"Loading model: {args.model_name}")

	processor = AutoImageProcessor.from_pretrained(args.model_name)
	model = Dinov2Model.from_pretrained(args.model_name)
	model.eval()
	model.to(args.device)

	storage_dtype = tensor_cast_dtype(args.dtype)

	failed_paths: List[Path] = []

	with torch.inference_mode():
		for batch_paths in tqdm(
			to_chunks(image_paths, args.batch_size),
			total=(len(image_paths) + args.batch_size - 1) // args.batch_size,
			desc="Embedding",
		):
			try:
				images = load_images(batch_paths)
			except (UnidentifiedImageError, OSError):
				for img_path in batch_paths:
					try:
						with Image.open(img_path) as img:
							_ = img.convert("RGB")
					except (UnidentifiedImageError, OSError):
						failed_paths.append(img_path)
				continue

			inputs = processor(images=images, return_tensors="pt")
			inputs = {k: v.to(args.device) for k, v in inputs.items()}

			outputs = model(**inputs, output_hidden_states=True)
			hidden_states = outputs.hidden_states

			for idx, image_path in enumerate(batch_paths):
				per_layer = []
				for layer_h in hidden_states:
					layer_tokens = layer_h[idx].detach().to(device="cpu", dtype=storage_dtype)
					if args.save_cls_only:
						layer_tokens = layer_tokens[0]
					per_layer.append(layer_tokens)

				save_obj = {
					"image_path": str(image_path),
					"model_name": args.model_name,
					"save_cls_only": args.save_cls_only,
					"layers": per_layer,
				}

				out_path = output_path_for(image_path, args.input_dir, args.output_dir)
				out_path.parent.mkdir(parents=True, exist_ok=True)
				torch.save(save_obj, out_path)

	if failed_paths:
		fail_log = args.output_dir / "failed_images.txt"
		fail_log.write_text("\n".join(str(p) for p in failed_paths) + "\n")
		print(f"Finished with {len(failed_paths)} failures. Logged to: {fail_log}")
	else:
		print("Finished without image decode failures.")

	print(f"Embeddings saved to: {args.output_dir}")


def main2():
	args = parse_args()

	if not args.input_dir.exists() or not args.input_dir.is_dir():
		raise ValueError(f"Input directory does not exist or is not a directory: {args.input_dir}")
	if args.batch_size <= 0:
		raise ValueError("batch-size must be a positive integer")

	args.output_dir.mkdir(parents=True, exist_ok=True)
	image_paths = find_images(args.input_dir)
	if not image_paths:
		print(f"No images found under: {args.input_dir}")
		return

	processor = AutoImageProcessor.from_pretrained(args.model_name)
	output_h5 = args.output_dir / "pixel_values.h5"
	storage_dtype = np.float32

	written = 0
	all_image_arr = None #np.zeros((len(image_paths), 3, processor.size[], processor.size["width"]), dtype=storage_dtype)
	written = set()

	with h5py.File(output_h5, "w") as h5f:
		for image_path in tqdm(
			image_paths
		):
			image = Image.open(image_path).convert("RGB")
			inputs = processor(images=image, return_tensors="pt")
			pixel_values = inputs["pixel_values"][0].cpu().numpy().astype(storage_dtype, copy=False)
			image_index = int(image_path.stem)
			if all_image_arr is None:
				all_image_arr = np.zeros((len(image_paths), *pixel_values.shape), dtype=storage_dtype)
			all_image_arr[image_index] = pixel_values
			if image_index not in written:
				written.add(image_index)
			else:
				raise ValueError(f"Duplicate image index detected: {image_index} from path: {image_path}")
			
		h5f.create_dataset("pixel_values", data=all_image_arr)
	assert len(written) == len(image_paths), f"Expected to write {len(image_paths)} unique images, but got {len(written)} unique indices."


	print(f"Saved {written} processed images to: {output_h5}")

if __name__ == "__main__":
	main2()
