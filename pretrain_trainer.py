import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
from utils.util import generate_mask
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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

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

    def train(self):
        best_loss = 10000
        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
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

                        # Next-patch prediction: output at t predicts input at t+1
                        # y: (B, n_ch, seq_len, patch_size), x: (B, n_ch, seq_len, patch_size)
                        pred = y[:, :, :-1, :]   # predictions from t=0..T-2
                        target = x[:, :, 1:, :]  # targets from t=1..T-1
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

                        masked_x = x[mask == 1]
                        masked_y = y[mask == 1]
                        mask_loss = self.criterion(masked_y, masked_x)
                        # recon_loss = self.criterion(y, x)
                        loss = mask_loss + sum(loss_dict.values())
                        # loss = sum(loss_dict.values())
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
                wandb.log({
                    **logs,
                    "epoch": epoch + 1,
                    "lr": self.optimizer_scheduler.get_last_lr()[0]
                })
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            if mean_loss < best_loss:
                # if self.params.model != 'OurModel' or (self.params.model == 'OurModel' and epoch % 10 == 0):
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print("Model saved at " + model_path)
                best_loss = mean_loss

    @torch.no_grad()
    def test(self, param):
        self.model._load_weights(param.foundation_dir, param)
        self.model.eval()

        for x in tqdm(self.data_loader, desc="Testing", mininterval=10):
            x = x.to(self.device) / 100
            y = self.model(x)
