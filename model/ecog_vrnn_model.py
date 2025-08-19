# ecog_vrnn_model_revised.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PVRNNCell(nn.Module):
    """論文の数式に厳密に基づくPVRNNセル"""
    
    def __init__(self, d_size, z_size, tau, level='local', device='cuda', higher_z_size=None):
        super().__init__()
        self.d_size = d_size
        self.z_size = z_size
        self.tau = tau
        self.level = level  # 'global', 'functional', 'local'
        self.device = device
        
        # ネットワーク重み（論文の記法に従う）
        self.d_to_z = nn.Linear(d_size, 2 * z_size, bias=False)  # 事前分布のパラメータ用
        self.d_to_h = nn.Linear(d_size, d_size, bias=False)
        self.z_to_h = nn.Linear(z_size, d_size, bias=False)
        
        # バイアス項
        self.bias = nn.Parameter(torch.zeros(d_size))
        
        # 上位層からの入力用（局所領域と機能ネットワークレベルのみ）
        if level in ['functional', 'local']:
            if level == 'functional':
                # グローバル状態 z^(3) から機能ネットワーク（d_size=15）へ
                in_dim = higher_z_size if higher_z_size is not None else 1
                self.higher_to_h = nn.Linear(in_dim, d_size, bias=False)
            else:  # local
                # 機能ネットワーク（z_size=3）から局所領域（d_size=15）へ
                self.higher_to_h = nn.Linear(3, d_size, bias=False)
    
    def compute_prior(self, d_prev):
        """
        事前分布 p(z_t|d_{t-1}) の計算
        式(1)-(3)に対応
        """
        # 式(2), (3): μとσの計算
        z_params = self.d_to_z(d_prev)
        mu_p_raw, log_sigma_p_raw = torch.chunk(z_params, 2, dim=-1)
        
        # tanh と exp の適用
        mu_p = torch.tanh(mu_p_raw)  # 式(2)
        sigma_p = torch.exp(log_sigma_p_raw)  # 式(3)
        
        return mu_p, sigma_p
    
    def sample_z_from_prior(self, mu_p, sigma_p):
        """
        事前分布からのサンプリング
        式(4)に対応
        """
        epsilon = torch.randn_like(mu_p)
        z = mu_p + sigma_p * epsilon
        return z
    
    def sample_z_from_posterior(self, mu_q, sigma_q):
        """
        事後分布からのサンプリング
        式(8)に対応
        """
        epsilon = torch.randn_like(mu_q)
        z = mu_q + sigma_q * epsilon
        return z
    
    def compute_mtrnn(self, h_prev, d_prev, z_current, higher_input=None):
        """
        MTRNNダイナミクスの計算
        式(9)-(10)に対応
        """
        # 基本入力の計算
        h_input = self.d_to_h(d_prev) + self.z_to_h(z_current) + self.bias
        
        # 上位層からの入力がある場合（式(9)の条件分岐）
        if higher_input is not None and hasattr(self, 'higher_to_h'):
            h_input = h_input + self.higher_to_h(higher_input)
        
        # 時定数を考慮した更新（式(9)）
        h = (1.0 / self.tau) * h_input + (1.0 - 1.0 / self.tau) * h_prev
        
        # 活性化関数の適用（式(10)）
        d = torch.tanh(h)
        
        return h, d

class AdaptiveA(nn.Module):
    """
    事後分布のための適応的内部状態 a_(s,t,i)
    式(5)-(7)に対応
    """
    def __init__(self, seq_len, z_size, level='local', device='cuda'):
        super().__init__()
        self.seq_len = seq_len
        self.z_size = z_size
        self.level = level
        self.device = device
        
        # a = [a^μ, a^σ] として初期化
        if level == 'global':
            # グローバルレベルは時間的に一定（式(5)の第1条件）
            self.a = nn.Parameter(torch.zeros(1, 2 * z_size, device=device))
        else:
            # 他のレベルは各時刻で異なる値
            self.a = nn.Parameter(torch.zeros(seq_len, 2 * z_size, device=device))
    
    def get_posterior_params(self, t):
        """時刻tにおける事後分布パラメータを取得"""
        if self.level == 'global':
            # グローバルレベルは時間に依存しない
            a_mu, a_sigma = torch.chunk(self.a, 2, dim=-1)
        else:
            a_mu, a_sigma = torch.chunk(self.a[t], 2, dim=-1)
        
        # 式(6), (7)の適用
        mu_q = torch.tanh(a_mu)
        sigma_q = torch.exp(a_sigma)
        
        return mu_q, sigma_q

class ECoGDigitalTwinRevised(nn.Module):
    """論文に忠実なECoGデジタルツインモデル"""
    
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.config = config
        
        # 設定から各サイズを取得
        self.global_z_size = config.get('global_z_size', 1)
        
        # Level 3: Global State (τ=100000)
        self.global_cell = PVRNNCell(
            d_size=2, z_size=self.global_z_size, tau=100000, level='global', device=device
        )
        
        # Level 2: Functional Network (τ=4)
        self.func_cell = PVRNNCell(
            d_size=15, z_size=3, tau=4, level='functional', device=device,
            higher_z_size=self.global_z_size
        )
        
        # Level 1: Local Regions (τ=2)
        self.local_cells = nn.ModuleList([
            PVRNNCell(d_size=15, z_size=10, tau=2, level='local', device=device)
            for _ in range(10)
        ])
        
        # 出力層（各領域2チャンネル）- 式(11)
        self.outputs = nn.ModuleList([
            nn.Linear(15, 2, bias=False) for _ in range(10)
        ])
        
        # メタプライア W^(l)
        self.W_global = config.get('W_global', 0.001)
        self.W_global_t0 = config.get('W_global_t0', 0.001)
        self.W_func = config.get('W_func', 0.001)
        self.W_local = config.get('W_local', 0.001)
        
    def initialize_states(self, batch_size):
        """状態の初期化"""
        # 決定論的状態（h, d）
        self.h_global = torch.zeros(batch_size, 2, device=self.device)
        self.d_global = torch.zeros(batch_size, 2, device=self.device)
        
        self.h_func = torch.zeros(batch_size, 15, device=self.device)
        self.d_func = torch.zeros(batch_size, 15, device=self.device)
        
        self.h_local = torch.zeros(batch_size, 10, 15, device=self.device)
        self.d_local = torch.zeros(batch_size, 10, 15, device=self.device)
        
    def forward(self, seq_len, batch_size, A_params=None, mode='train'):
        """
        前向き計算
        Args:
            seq_len: シーケンス長
            batch_size: バッチサイズ
            A_params: 事後分布パラメータ（訓練時は内部で作成、データ同化時は外部から）
            mode: 'train' or 'inference'
        """
        # 事後分布パラメータの準備
        if A_params is None:
            # 訓練時：内部でAパラメータを作成
            A_global = AdaptiveA(seq_len, self.global_z_size, level='global', device=self.device)
            A_func = AdaptiveA(seq_len, 3, level='functional', device=self.device)
            A_local = nn.ModuleList([
                AdaptiveA(seq_len, 10, level='local', device=self.device)
                for _ in range(10)
            ])
        else:
            A_global, A_func, A_local = A_params
        
        # 出力とKL損失の記録
        outputs = []
        kl_losses = {'global': [], 'func': [], 'local': []}
        
        # 状態の初期化
        self.initialize_states(batch_size)
        
        # 各時刻での処理
        for t in range(seq_len):
            # === Level 3: Global State ===
            if t == 0:
                # t=0では事前分布はN(0,1)
                mu_p_global = torch.zeros(batch_size, self.global_z_size, device=self.device)
                sigma_p_global = torch.ones(batch_size, self.global_z_size, device=self.device)
            else:
                # t>0では前時刻のdから計算
                mu_p_global, sigma_p_global = self.global_cell.compute_prior(self.d_global)
            
            # 事後分布からサンプリング
            mu_q_global, sigma_q_global = A_global.get_posterior_params(0)  # グローバルは時間一定
            mu_q_global = mu_q_global.expand(batch_size, -1)
            sigma_q_global = sigma_q_global.expand(batch_size, -1)
            
            z_global = self.global_cell.sample_z_from_posterior(mu_q_global, sigma_q_global)
            
            # KLダイバージェンス計算
            kl_global = self._compute_kl(
                mu_q_global, sigma_q_global, mu_p_global, sigma_p_global
            )
            
            # メタプライアによる重み付け
            if t == 0:
                weighted_kl_global = self.W_global_t0 * kl_global
            else:
                weighted_kl_global = self.W_global * kl_global
            
            kl_losses['global'].append(weighted_kl_global)
            
            # MTRNN更新（グローバルレベルは上位入力なし）
            self.h_global, self.d_global = self.global_cell.compute_mtrnn(
                self.h_global, self.d_global, z_global
            )
            
            # === Level 2: Functional Network ===
            if t == 0:
                mu_p_func = torch.zeros(batch_size, 3, device=self.device)
                sigma_p_func = torch.ones(batch_size, 3, device=self.device)
            else:
                mu_p_func, sigma_p_func = self.func_cell.compute_prior(self.d_func)
            
            # 事後分布
            mu_q_func, sigma_q_func = A_func.get_posterior_params(t)
            mu_q_func = mu_q_func.expand(batch_size, -1)
            sigma_q_func = sigma_q_func.expand(batch_size, -1)
            
            z_func = self.func_cell.sample_z_from_posterior(mu_q_func, sigma_q_func)
            
            # KLダイバージェンス
            kl_func = self._compute_kl(
                mu_q_func, sigma_q_func, mu_p_func, sigma_p_func
            )
            weighted_kl_func = self.W_func * kl_func
            kl_losses['func'].append(weighted_kl_func)
            
            # MTRNN更新（グローバルレベルのz_globalが入力）
            self.h_func, self.d_func = self.func_cell.compute_mtrnn(
                self.h_func, self.d_func, z_func, higher_input=z_global
            )
            
            # === Level 1: Local Regions ===
            region_outputs = []
            kl_local_total = 0
            
            # インプレース更新を避けるため、一時バッファに保持
            new_h_local = []
            new_d_local = []
            
            for i in range(10):
                if t == 0:
                    mu_p_local = torch.zeros(batch_size, 10, device=self.device)
                    sigma_p_local = torch.ones(batch_size, 10, device=self.device)
                else:
                    mu_p_local, sigma_p_local = self.local_cells[i].compute_prior(
                        self.d_local[:, i, :]
                    )
                
                # 事後分布
                mu_q_local, sigma_q_local = A_local[i].get_posterior_params(t)
                mu_q_local = mu_q_local.expand(batch_size, -1)
                sigma_q_local = sigma_q_local.expand(batch_size, -1)
                
                z_local = self.local_cells[i].sample_z_from_posterior(
                    mu_q_local, sigma_q_local
                )
                
                # KLダイバージェンス
                kl_local = self._compute_kl(
                    mu_q_local, sigma_q_local, mu_p_local, sigma_p_local
                )
                kl_local_total += kl_local
                
                # MTRNN更新（機能ネットワークのz_funcが入力）
                h_new, d_new = self.local_cells[i].compute_mtrnn(
                    self.h_local[:, i, :], 
                    self.d_local[:, i, :], 
                    z_local, 
                    higher_input=z_func
                )
                
                # インプレース代入を避ける
                new_h_local.append(h_new)
                new_d_local.append(d_new)
                
                # 出力生成（式(11)）
                output = torch.tanh(self.outputs[i](d_new))
                region_outputs.append(output)
            
            # ここでまとめて更新（インプレース回避）
            self.h_local = torch.stack(new_h_local, dim=1)
            self.d_local = torch.stack(new_d_local, dim=1)
            
            weighted_kl_local = self.W_local * kl_local_total
            kl_losses['local'].append(weighted_kl_local)
            
            # 全チャンネルを結合
            output_t = torch.cat(region_outputs, dim=-1)  # (batch, 20)
            outputs.append(output_t)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, 20)
        
        return outputs, kl_losses, (A_global, A_func, A_local)
    
    def _compute_kl(self, mu_q, sigma_q, mu_p, sigma_p):
        """KLダイバージェンスの計算"""
        eps = 1e-8
        kl = 0.5 * torch.sum(
            torch.log(sigma_p.pow(2) + eps) - torch.log(sigma_q.pow(2) + eps) - 1 +
            (mu_q - mu_p).pow(2) / (sigma_p.pow(2) + eps) +
            sigma_q.pow(2) / (sigma_p.pow(2) + eps)
        ) / mu_q.shape[-1]  # z次元で正規化
        return kl