import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class EEGConvBlock(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm1d(hidden_channels)
		self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm1d(out_channels)
		self.activation = nn.GELU()

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.activation(x)

		x = self.conv2(x)
		x = self.bn2(x)

		return self.activation(x + residual)


class CNNSemanticReadout(nn.Module):
	def __init__(self, n_ch: int, out_dim: int, seq_len: int, hidden_dim: int = 64, dropout: float = 0.1):
		super().__init__()
		self.seq_len = seq_len

		self.input_proj = nn.Sequential(
			nn.Conv1d(n_ch, hidden_dim, kernel_size=1, bias=False),
			nn.GELU(),
		)

		self.temporal_blocks = nn.ModuleList([
			nn.Sequential(
				nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1, bias=False),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
			),
			nn.Sequential(
				nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
			),
			nn.Sequential(
				nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4, bias=False),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
			),
			nn.Sequential(
				nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8, bias=False),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
			),
		])

		self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, seq_len))

		self.head = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim * seq_len, 1024),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(1024, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.input_proj(x)
		h = h + self.pos_embed
		for block in self.temporal_blocks:
			h = h + block(h)

		return self.head(h)


class CSBrainCNN(nn.Module):
	def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
				 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[]):
		super().__init__()
		self.in_dim = in_dim
		self.seq_len = seq_len
		self.d_model = d_model
		self.sorted_indices = sorted_indices
		self.contrastive_temperature = 0.07
		n_ch = len(sorted_indices) if len(sorted_indices) > 0 else 32
		self.n_ch = n_ch
		
		num_hidden_ch = 64
		self.proj_in = nn.Conv1d(32, num_hidden_ch, kernel_size=1)
		self.encoder = nn.ModuleList(
			[EEGConvBlock(in_channels=num_hidden_ch, hidden_channels=num_hidden_ch, out_channels=num_hidden_ch) for _ in range(n_layer)]
		)

		self.proj_out = nn.Conv1d(num_hidden_ch, 32, kernel_size=1)

		self.pretrained_image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()
		self.semantic_readout = CNNSemanticReadout(
			n_ch=n_ch,
			out_dim=d_model,
			seq_len=240,
			hidden_dim=64,
		)
		self.contrastive_proj = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.GELU(),
			nn.Linear(d_model, 768),
		)

		# self.apply(_weights_init)

		self.features_by_layer = []
		self.input_features = []

	@torch.inference_mode()
	def get_image_hidden_states(self, **image_encoder_inputs):
		outputs = self.pretrained_image_encoder(**image_encoder_inputs, output_hidden_states=True)
		hidden_states = outputs.hidden_states
		selected_hidden_states = []
		for layer_idx in [12]:
			selected_hidden_states.append(hidden_states[layer_idx][:, 0])
		selected_hidden_states = torch.stack(selected_hidden_states, dim=1)
		return selected_hidden_states

	def forward(self, batch, mask=None):
		x = batch['timeseries']

		bz, ch_num, seq_len, patch_size = x.shape
		if ch_num != self.n_ch:
			raise ValueError(f"Expected n_ch={self.n_ch}, got {ch_num}")
		emb = self.proj_in(x.contiguous().view(bz, ch_num, seq_len * patch_size))

		for layer in self.encoder:
			emb = layer(emb)
		out = self.proj_out(emb)

		contrastive_loss = {}
		if batch.get('image_encoder_inputs') is not None:
			image_hidden_states = self.get_image_hidden_states(**batch['image_encoder_inputs'])
			i_branch = 0

			pred_flatten = self.contrastive_proj(self.semantic_readout(out))
			image_hidden_states_branch = image_hidden_states[:, i_branch]
			image_flatten = image_hidden_states_branch.reshape(-1, image_hidden_states_branch.size(-1))

			pred_norm = F.normalize(pred_flatten, dim=-1)
			image_norm = F.normalize(image_flatten, dim=-1)

			logits = torch.matmul(pred_norm, image_norm.t()) / self.contrastive_temperature
			targets = torch.arange(logits.size(0), device=logits.device)

			loss_p2i = F.cross_entropy(logits, targets)
			loss_i2p = F.cross_entropy(logits.t(), targets)
			branch_loss = 0.5 * (loss_p2i + loss_i2p)

			contrastive_loss[f"contrastive_loss_{i_branch}"] = (1.0, branch_loss)
			with torch.no_grad():
				pred_to_img = logits.argmax(dim=1)
				img_to_pred = logits.argmax(dim=0)

				acc_p2i = (pred_to_img == targets).float().mean()
				acc_i2p = (img_to_pred == targets).float().mean()
				acc = 0.5 * (acc_p2i + acc_i2p)

			contrastive_loss[f"contrastive_acc_{i_branch}"] = acc
			

		return x, {
			**contrastive_loss,
		}

	def training_step(self, *args, **kwargs):
		return self.forward(*args, **kwargs)


def _weights_init(m):
	if isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
	if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
	elif isinstance(m, nn.BatchNorm1d):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)

