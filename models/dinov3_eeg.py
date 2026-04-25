"""
DINOv3/v2 EEG classifier via EEG-to-image transformation.

Two image modes are supported:

``image_mode='raw'`` (default, matches the multi-teacher-distillation paper):
    EEG (C, T) is treated as a single-channel image with H=C, W=T.
    Per-sample min-max to [0, 1], replicated to RGB, resized to the
    backbone's crop size, then ImageNet-normalised. This preserves the
    inter-channel amplitude structure that motor-imagery decoding relies on.

``image_mode='spectrogram'``:
    Per-channel log-magnitude STFTs tiled into a sqrt(C)xsqrt(C) grid,
    then resized + ImageNet-normalised. Normalises each channel-spectrogram
    independently so it erases amplitude differences across electrodes --
    useful for stationarity-invariant tasks, poor for motor imagery.

Naming note: the vision backbone is stored as ``self.dino_encoder`` (not
``self.backbone``) so that ``finetune_trainer`` does *not* flip
``requires_grad`` on its parameters — that loop matches on the substring
"backbone" and would otherwise un-freeze the DINO weights we want to keep
frozen while training only LoRA + classifier.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


class EEGRawImage(nn.Module):
    """EEG (B, C, T) -> RGB image (B, 3, crop_size, crop_size) via direct 2D layout.

    H = n_channels, W = n_timesteps. Per-sample min-max to [0, 1], bicubic
    resize (antialiased) to the backbone's crop size, replicate to RGB,
    then ImageNet normalise with the backbone's own mean/std.
    """

    def __init__(
        self,
        crop_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ):
        super().__init__()
        self.crop_size = crop_size
        self.register_buffer(
            "im_mean", torch.tensor(list(image_mean)).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "im_std", torch.tensor(list(image_std)).view(1, 3, 1, 1), persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        flat = x.reshape(B, -1)
        mn = flat.min(dim=-1, keepdim=True).values.view(B, 1, 1, 1)
        mx = flat.max(dim=-1, keepdim=True).values.view(B, 1, 1, 1)
        rng = (mx - mn).clamp(min=1e-6)

        img = x.unsqueeze(1)  # (B, 1, C, T)
        img = (img - mn) / rng

        img = F.interpolate(
            img, size=(self.crop_size, self.crop_size),
            mode='bicubic', align_corners=False, antialias=True,
        )
        img = img.clamp(0.0, 1.0).expand(-1, 3, -1, -1)
        return (img - self.im_mean) / self.im_std


class EEGSpectrogramImage(nn.Module):
    """EEG (B, C, T) -> RGB image (B, 3, crop_size, crop_size) via tiled per-channel STFTs.

    Mirrors the DINO image processor: bicubic-resize shortest edge to
    ``resize_size`` (with antialiasing), center-crop to ``crop_size``, replicate
    the grayscale channel to RGB, then normalise with the encoder's own
    ``image_mean`` / ``image_std``.
    """

    def __init__(
        self,
        crop_size: int,
        resize_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: Optional[int] = None,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "im_mean", torch.tensor(list(image_mean)).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "im_std", torch.tensor(list(image_std)).view(1, 3, 1, 1), persistent=False,
        )

    @staticmethod
    def _grid_shape(n_channels: int, grid_rows, grid_cols):
        if grid_rows is not None and grid_cols is not None:
            return grid_rows, grid_cols
        side = int(math.ceil(math.sqrt(n_channels)))
        return side, side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        window = self.window.to(dtype=x.dtype)

        spec = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B*C, F, Ts)
        log_mag = torch.log1p(spec.abs())

        flat = log_mag.reshape(B, C, -1)
        mean = flat.mean(dim=-1, keepdim=True).unsqueeze(-1)
        std = flat.std(dim=-1, keepdim=True).clamp(min=1e-5).unsqueeze(-1)
        log_mag = torch.sigmoid((log_mag.view(B, C, *log_mag.shape[-2:]) - mean) / std)

        F_, Ts = log_mag.shape[-2:]
        R, Co = self._grid_shape(C, self.grid_rows, self.grid_cols)
        if R * Co > C:
            log_mag = F.pad(log_mag, (0, 0, 0, 0, 0, R * Co - C))

        grid = log_mag.view(B, R, Co, F_, Ts).permute(0, 1, 3, 2, 4).contiguous()
        grid = grid.view(B, 1, R * F_, Co * Ts)

        # Match the DINO image processor: shortest-edge bicubic resize (antialiased)
        # -> center crop -> clamp to [0, 1] -> replicate to RGB -> normalise.
        H, W = grid.shape[-2:]
        if H <= W:
            new_h = self.resize_size
            new_w = int(round(W * self.resize_size / H))
        else:
            new_w = self.resize_size
            new_h = int(round(H * self.resize_size / W))
        grid = F.interpolate(
            grid, size=(new_h, new_w),
            mode='bicubic', align_corners=False, antialias=True,
        )
        top = (new_h - self.crop_size) // 2
        left = (new_w - self.crop_size) // 2
        grid = grid[..., top:top + self.crop_size, left:left + self.crop_size]
        grid = grid.clamp(0.0, 1.0).expand(-1, 3, -1, -1)
        return (grid - self.im_mean) / self.im_std


def _resolve_preprocessing(vision_encoder: str):
    """Return (crop_size, resize_size, image_mean, image_std) from the DINO processor.

    DINOv2 has size={'shortest_edge':256} + crop_size={'height':224,'width':224};
    DINOv3 typically has size={'height':224,'width':224} with do_center_crop=False.
    This normalises both into a (crop_size, resize_size) pair we can apply
    with ``F.interpolate`` + a slice.
    """
    proc = AutoImageProcessor.from_pretrained(vision_encoder)
    size = proc.size
    cs = getattr(proc, 'crop_size', None)
    do_crop = getattr(proc, 'do_center_crop', False)

    if do_crop and isinstance(cs, dict) and 'height' in cs:
        crop = int(cs['height'])
        if isinstance(size, dict) and 'shortest_edge' in size:
            resize = int(size['shortest_edge'])
        elif isinstance(size, dict) and 'height' in size:
            resize = int(size['height'])
        else:
            resize = crop
    else:
        if isinstance(size, dict) and 'height' in size:
            crop = int(size['height'])
        elif isinstance(size, dict) and 'shortest_edge' in size:
            crop = int(size['shortest_edge'])
        else:
            crop = 224
        resize = crop
    return crop, resize, list(proc.image_mean), list(proc.image_std)


class Model(nn.Module):
    """DINOv3/v2-based EEG classifier.

    Expected batch: {'x': (B, C, N, d) or (B, C, T), 'y': labels, ...}.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        vision_encoder = getattr(params, 'vision_encoder', 'facebook/dinov2-base')
        self.dino_encoder = AutoModel.from_pretrained(vision_encoder)
        hidden = self.dino_encoder.config.hidden_size

        crop_size, resize_size, image_mean, image_std = _resolve_preprocessing(vision_encoder)
        if getattr(params, 'image_size', 0):
            crop_size = int(params.image_size)
            resize_size = crop_size

        image_mode = str(getattr(params, 'image_mode', 'raw')).lower()
        if image_mode == 'raw':
            self.eeg_to_image = EEGRawImage(
                crop_size=crop_size,
                image_mean=image_mean,
                image_std=image_std,
            )
        elif image_mode == 'spectrogram':
            self.eeg_to_image = EEGSpectrogramImage(
                crop_size=crop_size,
                resize_size=resize_size,
                image_mean=image_mean,
                image_std=image_std,
                n_fft=int(getattr(params, 'stft_n_fft', 64)),
                hop_length=int(getattr(params, 'stft_hop_length', 16)),
            )
        else:
            raise ValueError(f"image_mode must be 'raw' or 'spectrogram', got {image_mode!r}")

        use_lora = bool(getattr(params, 'use_lora', False))
        freeze_backbone = bool(getattr(params, 'freeze_backbone', False))

        if use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=int(getattr(params, 'lora_rank', 8)),
                lora_alpha=int(getattr(params, 'lora_alpha', 16)),
                lora_dropout=float(getattr(params, 'lora_dropout', 0.0)),
                bias='none',
                target_modules=['query', 'key', 'value'],
            )
            self.dino_encoder = get_peft_model(self.dino_encoder, cfg)
        elif freeze_backbone:
            for p in self.dino_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(params.dropout),
            nn.Linear(hidden, params.num_of_classes),
        )

    def forward(self, batch):
        x = batch['x']
        if x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        images = self.eeg_to_image(x)
        out = self.dino_encoder(pixel_values=images)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)
