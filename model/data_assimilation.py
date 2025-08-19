# data_assimilation_revised.py
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from ecog_vrnn_model import AdaptiveA
import torch.nn as nn

class DataAssimilationRevised:
    """論文に忠実なデータ同化実装"""
    
    def __init__(self, model, config, device='cuda'):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device
        
        # データ同化パラメータ
        self.window_size = config.get('window_size', 500)
        self.n_updates = config.get('n_updates', 100)
        self.n_updates_initial = config.get('n_updates_initial', 50)
        self.lr = config.get('da_learning_rate', 0.001)
        self.da_stride = config.get('da_stride', 1) # configからda_strideを読み込み
        
        # モデルのグローバルz次元に追従
        self.global_z_size = getattr(model, 'global_z_size', 1)
        
    def initialize_posterior_params(self, window_size):
        """事後分布パラメータの初期化"""
        # 各レベルのAdaptiveAを作成
        A_global = AdaptiveA(window_size, self.global_z_size, level='global', device=self.device)
        A_func = AdaptiveA(window_size, 3, level='functional', device=self.device)
        A_local = nn.ModuleList([
            AdaptiveA(window_size, 10, level='local', device=self.device)
            for _ in range(10)
        ])
        
        return A_global, A_func, A_local
    
    def update_window(self, observation_window, A_params, initial=False):
        """
        時間窓内での事後分布更新
        式(13)-(14)に基づく
        戻り値: (free_energy: float, predictions: Tensor[1, window, 20])
        """
        A_global, A_func, A_local = A_params
        
        # 更新回数の設定
        n_updates = self.n_updates_initial if initial else self.n_updates
        
        # 最適化対象パラメータ
        params = [{'params': A_global.parameters()}]
        params.append({'params': A_func.parameters()})
        for a_local in A_local:
            params.append({'params': a_local.parameters()})
        
        optimizer = optim.Adam(params, lr=self.lr)
        
        # 初期時刻では事前分布で初期化
        if initial:
            with torch.no_grad():
                # t=0はゼロ（N(0,1)に対応）
                A_global.a.data.zero_()
                A_func.a.data[0].zero_()
                for a_local in A_local:
                    a_local.a.data[0].zero_()
        
        last_predictions = None
        
        # 更新ループ
        for update_idx in range(n_updates):
            # 状態リセット
            self.model.initialize_states(batch_size=1)
            
            # 時間窓内での順伝播
            predictions, kl_losses, _ = self.model(
                seq_len=self.window_size,
                batch_size=1,
                A_params=(A_global, A_func, A_local),
                mode='inference'
            )
            last_predictions = predictions  # (1, window, 20)
            
            # 自由エネルギー計算（式(14)）
            observation_window_batch = observation_window.unsqueeze(0)  # (1, window_size, 20)
            
            recon_error = 0.5 * torch.sum(
                (predictions - observation_window_batch) ** 2
            ) / predictions.shape[-1]
            
            kl_total = 0
            for level in ['global', 'func', 'local']:
                if level in kl_losses:
                    kl_total += sum(kl_losses[level])
            
            free_energy = recon_error + kl_total
            
            # 最適化
            optimizer.zero_grad()
            free_energy.backward()
            optimizer.step()
            
            if update_idx % 10 == 0:
                print(f"Update {update_idx}: FE={free_energy.item():.4f}")
        
        return free_energy.item(), last_predictions.detach()
    
    def slide_window_params(self, A_params):
        """窓をスライドする際のパラメータ更新 (da_strideを考慮)"""
        A_global, A_func, A_local = A_params
        
        with torch.no_grad():
            # グローバルレベルは時間一定なので変更なし
            # ただし、A_global.aは(1, 2*z_size)なので、スライドは不要

            # 機能ネットワークレベル
            if self.window_size >= self.da_stride: # スライド可能な場合
                A_func.a.data[:-self.da_stride] = A_func.a.data[self.da_stride:].clone()
                # 最後の da_stride 個の要素を再初期化
                A_func.a.data[-self.da_stride:] = torch.randn_like(A_func.a.data[-self.da_stride:]) * 0.1
            else: # da_strideがwindow_sizeより大きい場合、全てを再初期化
                A_func.a.data = torch.randn_like(A_func.a.data) * 0.1

            # 局所領域レベル
            for a_local in A_local:
                if self.window_size >= self.da_stride: # スライド可能な場合
                    a_local.a.data[:-self.da_stride] = a_local.a.data[self.da_stride:].clone()
                    # 最後の da_stride 個の要素を再初期化
                    a_local.a.data[-self.da_stride:] = torch.randn_like(a_local.a.data[-self.da_stride:]) * 0.1
                else:
                    a_local.a.data = torch.randn_like(a_local.a.data) * 0.1
    
    def assimilate(self, ecog_sequence):
        """
        完全なデータ同化プロセス
        戻り値: (z3_trajectory: np.ndarray[time, 1|2], predicted_ecog: np.ndarray[time, 20])
        """
        ecog_sequence = torch.FloatTensor(ecog_sequence).to(self.device)
        seq_len = ecog_sequence.shape[0]
        
        # 事後分布パラメータの初期化
        # AdaptiveAのサイズは window_size で初期化される
        A_params = self.initialize_posterior_params(self.window_size)
        
        # 潜在状態の履歴
        z3_trajectory = []
        # 予測系列（各ステップで追加される新規予測部分を格納）
        predicted_segments = []
        
        # 初期窓の処理
        print("Processing initial window...")
        initial_window = ecog_sequence[:self.window_size]
        _, preds_init = self.update_window(initial_window, A_params, initial=True)
        preds_init_np = preds_init.squeeze(0).detach().cpu().numpy()  # (window_size, 20)
        
        # 最初の予測は全ウィンドウ分を格納
        predicted_segments.append(preds_init_np)
        
        # 初期状態の抽出（初期ウィンドウの最後の z^3 を記録）
        with torch.no_grad():
            self.model.initialize_states(batch_size=1)
            _, _, _ = self.model(
                seq_len=self.window_size,
                batch_size=1,
                A_params=A_params,
                mode='inference'
            )
            z3_state = self.model.d_global.cpu().numpy()
            z3_trajectory.append(z3_state)
        
        # スライディングウィンドウ
        # ループは da_stride ずつ進む。t は現在のウィンドウの開始インデックス
        # 最後の有効なウィンドウの開始インデックスは `seq_len - self.window_size`
        # range の終点は `seq_len - self.window_size + self.da_stride` にすることで、
        # `t` が `seq_len - self.window_size` まで含まれるようにする。
        for t in tqdm(range(self.da_stride, seq_len - self.window_size + self.da_stride, self.da_stride), 
                     desc="Data Assimilation"):
            
            # `t + self.window_size` が `seq_len` を超える場合は、
            # 最後の部分（残りのデータ）でウィンドウを形成する
            current_window_end = min(t + self.window_size, seq_len)
            current_window_actual_len = current_window_end - t
            
            current_window = ecog_sequence[t:current_window_end]

            # AdaptiveA パラメータをスライド。t-stride の予測に基づき t の状態を初期化
            # A_paramsは `window_size` の長さを持ち、これを `da_stride` だけスライド
            self.slide_window_params(A_params)

            # 新しい窓での更新（window_size は変わらないので、不足分はパディングされる想定か？）
            # 現実装では常に `self.window_size` で順伝播が呼ばれるため、
            # `current_window_actual_len` が `self.window_size` より短い場合は問題になる。
            # しかし、`RealtimeTestDataset` が固定長 (4000) であり、
            # `t + self.window_size` が `seq_len` を超えることは最後のイテレーション以外ではないため、
            # 基本的には `current_window` の長さは `window_size` となる。
            # もし最後のウィンドウが短い場合は、モデルが自動的にパディング等を処理すると仮定する。
            
            _, preds_win = self.update_window(current_window, A_params, initial=False)

            # 予測されたウィンドウ `preds_win` のうち、新しく予測された部分 (最後の `da_stride` サンプル) を取得
            # preds_win は (1, self.window_size, 20) の形状になる。
            # したがって、常に `preds_win[:, -self.da_stride:, :]` を取る。
            new_predicted_segment = preds_win.squeeze(0).detach().cpu().numpy()[-self.da_stride:]
            predicted_segments.append(new_predicted_segment)
            
            # 潜在状態の抽出（現在のウィンドウの最後の z^3 を記録）
            with torch.no_grad():
                self.model.initialize_states(batch_size=1)
                _, _, _ = self.model(
                    seq_len=self.window_size, # ここは常に window_size で呼ぶ
                    batch_size=1,
                    A_params=A_params,
                    mode='inference'
                )
                z3_state = self.model.d_global.cpu().numpy()
                z3_trajectory.append(z3_state)
        
        # 全ての予測セグメントを結合し、元の系列長にクリップ
        # `predicted_segments` の最初の要素は `window_size` 長のセグメント
        # それ以降の要素は各 `da_stride` 長のセグメント
        # 結合される全体の長さは `window_size + (num_loop_iterations * da_stride)`
        # これを `seq_len` にクリップする必要がある。
        predicted_ecog = np.concatenate(predicted_segments, axis=0)[:seq_len]

        return np.array(z3_trajectory), predicted_ecog