# training_revised.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class ECoGTrainerRevised:
    """論文の数式に厳密に基づく訓練実装"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Rectified Adam optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        
        self.train_history = {
            'loss': [],
            'recon_error': [],
            'kl_loss': []
        }
        
    def compute_free_energy(self, predictions, targets, kl_losses):
        """
        自由エネルギーの計算（式(12)）
        F_t = 1/2 ||x_t - x̂_t||^2 + Σ W^(l) * DKL[q||p]
        """
        batch_size, seq_len, n_channels = predictions.shape
        
        # 予測誤差項（チャンネル数で正規化）
        reconstruction_error = 0.5 * torch.sum((targets - predictions) ** 2) / n_channels
        
        # KL項（既にメタプライアで重み付け済み）
        kl_total = 0
        for level in ['global', 'func', 'local']:
            if level in kl_losses:
                kl_total += sum(kl_losses[level])
        
        # 全体の自由エネルギー
        free_energy = reconstruction_error + kl_total
        
        return free_energy, reconstruction_error, kl_total
    
    def train_epoch(self, train_loader, epoch):
        """1エポックの訓練"""
        self.model.train()
        epoch_stats = {
            'loss': [],
            'recon_error': [],
            'kl_loss': []
        }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            ecog_data = batch['ecog'].to(self.device)
            batch_size, seq_len, n_channels = ecog_data.shape
            
            # 順伝播
            predictions, kl_losses, A_params = self.model(
                seq_len=seq_len,
                batch_size=batch_size,
                mode='train'
            )
            
            # 自由エネルギー計算
            free_energy, recon_error, kl_total = self.compute_free_energy(
                predictions, ecog_data, kl_losses
            )
            
            # 逆伝播
            self.optimizer.zero_grad()
            free_energy.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Aパラメータの勾配もクリッピング
            for A in A_params:
                if isinstance(A, nn.ModuleList):
                    for a in A:
                        torch.nn.utils.clip_grad_norm_(a.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(A.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計記録
            epoch_stats['loss'].append(free_energy.item())
            epoch_stats['recon_error'].append(recon_error.item())
            epoch_stats['kl_loss'].append(kl_total.item())
            
            # 定期的な表示
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: FE={free_energy.item():.4f}, "
                      f"Recon={recon_error.item():.4f}, KL={kl_total.item():.4f}")
        
        # エポック統計
        for key in epoch_stats:
            self.train_history[key].append(np.mean(epoch_stats[key]))
        
        return {k: np.mean(v) for k, v in epoch_stats.items()}
