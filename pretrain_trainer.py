import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
from utils.util import generate_mask, save_pretrain_checkpoint, build_muon_optimizer
import os
import wandb
import time
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.adversarial import SessionDiscriminator, construct_session_pairs

def to_device(x, device):
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x

class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        # Only optimise parameters that require gradients (teacher params
        # in DINOEEGModel are frozen).
        optimizer_name = getattr(self.params, 'optimizer', 'AdamW')
        if optimizer_name == 'Muon':
            self.optimizer = build_muon_optimizer(
                self.model,
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
                muon_lr=getattr(self.params, 'muon_lr', 0.02),
            )
        else:
            trainable_params = [p for p in self.model.parameters()
                                if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.params.lr,
                                               weight_decay=self.params.weight_decay)
        # Cache initial LRs so the linear warmup below scales each
        # param group proportionally rather than slamming them to a
        # single value (Muon + AdamW have different base LRs).
        self._base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

        # --- Adversarial session discriminator ---
        self.adversarial_weight = getattr(self.params, 'adversarial_weight', 0.0)
        if self.adversarial_weight > 0:
            d_model = getattr(self.params, 'd_model', self.params.in_dim)
            self.discriminator = SessionDiscriminator(d_model=d_model).to(self.device)
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=1e-3
            )

        if self.params.lr_scheduler == 'CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler == 'ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler == 'StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler == 'MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler == 'CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )

        print(self.model)

    def _get_dino_model(self):
        """Return the underlying DINOEEGModel if wrapped, else None."""
        m = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(m, 'update_teacher'):
            return m
        return None

    def train(self):
        best_loss = 10000
        total_iters = self.params.epochs * self.data_length
        teacher_temp_warmup_iters = (
            getattr(self.params, 'teacher_temp_warmup_epochs', 30)
            * self.data_length
        )
        lr_warmup_iters = getattr(self.params, 'lr_warmup_iters', 0)
        base_lrs = self._base_lrs

        for epoch in range(self.params.epochs):
            # Let world-model wrappers ramp their prediction weight.
            m = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(m, 'current_epoch'):
                m.current_epoch.fill_(float(epoch))
            losses = []
            for batch_idx, x in enumerate(tqdm(self.data_loader, mininterval=10)):
                self.optimizer.zero_grad()
                if True:#self.params.model != 'CSBrain':
                    if isinstance(x, dict):
                        batch = x
                        batch = to_device(batch, self.device)
                        x = batch['timeseries']
                        x = x / 100
                        batch['timeseries'] = x
                    else:
                        batch = {'timeseries': x.to(self.device) / 100}
                    if getattr(self.params, 'causal', False):
                        # Causal next-patch prediction: no masking needed
                        out = self.model.training_step(batch, mask=None)

                        loss_dict = {}
                        logs = {}
                        if isinstance(out, tuple):
                            y, info = out
                            for key, value in info.items():
                                if 'loss' in key:
                                    coef, lss = value
                                    loss_dict[key] = coef * lss
                                    logs[key] = lss.data.cpu().numpy()
                                elif 'acc' in key:
                                    logs[key] = value.data.cpu().numpy()
                                elif key.startswith('diag_'):
                                    logs[key] = (
                                        value.detach().cpu().numpy()
                                        if torch.is_tensor(value) else value)

                        if info.get('dino_mode', False):
                            # DINO replaces the primary causal loss
                            loss = sum(loss_dict.values())
                        else:
                            pred = y[:, :, :-1, :]
                            target = x[:, :, 1:, :]
                            causal_loss = self.criterion(pred, target)
                            loss = causal_loss + sum(loss_dict.values())
                            logs["causal_loss"] = causal_loss.data.cpu().numpy()
                    elif self.params.need_mask:
                        bz, ch_num, patch_num, patch_size = x.shape
                        mask = generate_mask(
                            bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                        )
                        out = self.model.training_step(batch, mask=mask)

                        loss_dict = {}
                        logs = {}
                        if isinstance(out, tuple):
                            y, info = out
                            for key, value in info.items():
                                if 'loss' in key:
                                    coef, lss = value
                                    loss_dict[key] = coef * lss
                                    logs[key] = lss.data.cpu().numpy()
                                elif 'acc' in key:
                                    logs[key] = value.data.cpu().numpy()
                                elif key.startswith('diag_'):
                                    logs[key] = (
                                        value.detach().cpu().numpy()
                                        if torch.is_tensor(value) else value)
                        if info.get('dino_mode', False):
                            # DINO + iBOT replace the masked reconstruction loss
                            loss = sum(loss_dict.values())
                        else:
                            masked_x = x[mask == 1]
                            masked_y = y[mask == 1]
                            mask_loss = self.criterion(masked_y, masked_x)
                            loss = mask_loss + sum(loss_dict.values())
                            logs["mask_loss"] = mask_loss.data.cpu().numpy()
                    else:
                        out = self.model.training_step(batch, mask=None)

                        loss_dict = {}
                        logs = {}
                        if isinstance(out, tuple):
                            y, info = out
                            for key, value in info.items():
                                if 'loss' in key:
                                    coef, lss = value
                                    loss_dict[key] = coef * lss
                                    logs[key] = lss.data.cpu().numpy()
                                elif 'acc' in key:
                                    logs[key] = value.data.cpu().numpy()
                                elif key.startswith('diag_'):
                                    logs[key] = (
                                        value.detach().cpu().numpy()
                                        if torch.is_tensor(value) else value)

                        loss = sum(loss_dict.values())
                else:
                    if isinstance(x, dict):
                        x = x['timeseries']
                    x = x.to(self.device) / 100
                    if self.params.need_mask:
                        bz, ch_num, patch_num, patch_size = x.shape
                        mask = generate_mask(
                            bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                        )
                        y = self.model(x, mask=mask)
                        masked_x = x[mask == 1]
                        masked_y = y[mask == 1]
                        loss = self.criterion(masked_y, masked_x)
                    else:
                        y = self.model(x)
                        loss = self.criterion(y, x)
                    logs = {
                        "mask_loss": loss.data.cpu().numpy(),
                    }
                # --- Adversarial session-agnostic training ---
                if self.adversarial_weight > 0 and isinstance(out, tuple):
                    global_rep = info.get('global_rep')
                    session_ids = batch.get('session_id')
                    if global_rep is not None and session_ids is not None:
                        idx_a, idx_b, pair_labels = construct_session_pairs(
                            session_ids, max_pairs=128
                        )
                        if len(idx_a) > 0:
                            pair_labels = pair_labels.to(self.device)
                            idx_a = idx_a.to(self.device)
                            idx_b = idx_b.to(self.device)

                            # Step 1: train discriminator (detached encoder)
                            rep_det = global_rep.detach()
                            disc_logits = self.discriminator(
                                rep_det[idx_a], rep_det[idx_b]
                            )
                            disc_loss = F.binary_cross_entropy_with_logits(
                                disc_logits, pair_labels
                            )
                            self.disc_optimizer.zero_grad()
                            disc_loss.backward()
                            self.disc_optimizer.step()

                            # Step 2: adversarial loss for encoder (fool disc)
                            for p in self.discriminator.parameters():
                                p.requires_grad = False
                            enc_logits = self.discriminator(
                                global_rep[idx_a], global_rep[idx_b]
                            )
                            # flip labels: encoder should make disc wrong
                            adv_loss = F.binary_cross_entropy_with_logits(
                                enc_logits, 1.0 - pair_labels
                            )
                            loss = loss + self.adversarial_weight * adv_loss
                            for p in self.discriminator.parameters():
                                p.requires_grad = True

                            with torch.no_grad():
                                disc_acc = ((disc_logits > 0).float() == pair_labels).float().mean()
                            logs["disc_loss"] = disc_loss.data.cpu().numpy()
                            logs["disc_acc"] = disc_acc.data.cpu().numpy()
                            logs["adv_loss"] = adv_loss.data.cpu().numpy()

                loss.backward()
                if self.params.clip_value > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                logs["grad_norm"] = grad_norm.data.cpu().numpy()

                # --- Linear LR warmup (overrides scheduler during warmup) ---
                iteration = epoch * self.data_length + batch_idx
                if lr_warmup_iters > 0 and iteration < lr_warmup_iters:
                    factor = (iteration + 1) / lr_warmup_iters
                    for pg, base in zip(self.optimizer.param_groups, base_lrs):
                        pg['lr'] = base * factor

                self.optimizer.step()
                # Only step the cosine scheduler after warmup ends so the
                # schedule isn't consumed during the warmup phase.
                if iteration >= lr_warmup_iters:
                    self.optimizer_scheduler.step()

                # --- World-model EMA target encoder update ---
                wm_model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(wm_model, 'update_target_encoder'):
                    wm_model.update_target_encoder()

                # --- DINOv2 EMA teacher update & schedule step ---
                dino_model = self._get_dino_model()
                if dino_model is not None:
                    momentum = dino_model.get_ema_momentum(
                        iteration, total_iters)
                    dino_model.update_teacher(momentum)
                    dino_model.update_schedules(
                        iteration, total_iters, teacher_temp_warmup_iters)
                    dino_model.maybe_unfreeze_last_layer(iteration)
                    logs["ema_momentum"] = momentum
                    logs["teacher_temp"] = dino_model.dino_criterion.teacher_temp

                losses.append(loss.data.cpu().numpy())
                wandb.log({
                    **logs,
                    "epoch": epoch + 1,
                    "lr": self.optimizer_scheduler.get_last_lr()[0]
                })
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            if True: #mean_loss < best_loss:
                # if self.params.model != 'OurModel' or (self.params.model == 'OurModel' and epoch % 10 == 0):
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                save_pretrain_checkpoint(model_path, self.model, self.params)
                print("Model saved at " + model_path)
                best_loss = mean_loss

    @torch.no_grad()
    def test(self, param):
        self.model._load_weights(param.foundation_dir, param)
        self.model.eval()

        for x in tqdm(self.data_loader, desc="Testing", mininterval=10):
            x = x.to(self.device) / 100
            y = self.model(x)
