# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import torch
import seaborn as sns

class ECoGEvaluator:
    """評価と可視化のためのツール"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def extract_z3_trajectory(self, data_assimilation, test_sequence):
        """データ同化からz^(3)の軌跡を抽出"""
        z3_trajectory = []
        
        for latent_states in data_assimilation.assimilate(test_sequence):
            # Level 3 (Global State) の潜在状態を抽出
            z3 = latent_states['z_global'].cpu().numpy()
            z3_trajectory.append(z3)
            
        return np.array(z3_trajectory)
    
    def evaluate_clustering(self, z3_states, labels):
        """z^(3)のクラスタリング評価（論文の手法）"""
        # 2次元のz^(3)に対してクラスタリング
        z3_flat = z3_states.reshape(-1, z3_states.shape[-1])
        
        # シルエットスコアの計算
        silhouette_avg = silhouette_score(z3_flat, labels)
        
        # k-NNによる分類精度
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(z3_flat, labels)
        accuracy = knn.score(z3_flat, labels)
        
        return {
            'silhouette_score': silhouette_avg,
            'knn_accuracy': accuracy
        }
    
    def plot_z3_trajectory(self, z3_trajectory, transition_point=None):
        """z^(3)の軌跡をプロット（論文Figure 2b風）"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 2次元のz^(3)をプロット
        if z3_trajectory.shape[-1] == 2:
            x = z3_trajectory[:, 0, 0]
            y = z3_trajectory[:, 0, 1]
        else:
            # 1次元の場合は時間軸を追加
            x = np.arange(len(z3_trajectory))
            y = z3_trajectory[:, 0, 0]
        
        # 軌跡をプロット
        ax.plot(x, y, 'k-', alpha=0.5, linewidth=1)
        ax.scatter(x, y, c=np.arange(len(x)), cmap='viridis', s=20)
        
        # 遷移点をマーク
        if transition_point is not None:
            ax.axvline(x=transition_point, color='r', linestyle='--', 
                      label='Anesthesia → Awake')
        
        ax.set_xlabel('z³ dimension 1')
        ax.set_ylabel('z³ dimension 2')
        ax.set_title('z³ Trajectory during Data Assimilation')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_consciousness_clusters(self, z3_train_anesthesia, z3_train_awake, 
                                   z3_test=None):
        """意識状態のクラスタをプロット"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 訓練データのクラスタ
        ax.scatter(z3_train_anesthesia[:, 0], z3_train_anesthesia[:, 1], 
                  c='red', alpha=0.5, label='Anesthetized (Training)')
        ax.scatter(z3_train_awake[:, 0], z3_train_awake[:, 1], 
                  c='cyan', alpha=0.5, label='Awake (Training)')
        
        # テストデータの軌跡
        if z3_test is not None:
            ax.plot(z3_test[:, 0], z3_test[:, 1], 'gray', 
                   alpha=0.7, linewidth=2, label='Test Trajectory')
        
        ax.set_xlabel('z³ μ #1')
        ax.set_ylabel('z³ μ #2')
        ax.set_title('Consciousness State Clusters in z³ Space')
        ax.legend()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        
        plt.tight_layout()
        return fig
    
    def compute_power_spectrum(self, ecog_signals, sampling_rate=1000):
        """パワースペクトル密度の計算（論文の検証用）"""
        from scipy import signal
        
        freqs, psd = signal.welch(ecog_signals, fs=sampling_rate, 
                                 nperseg=2000, axis=0)
        
        return freqs, psd
    
    def plot_power_spectrum_comparison(self, observed_ecog, predicted_ecog, 
                                     brain_regions):
        """観測と予測のパワースペクトル比較"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, region in enumerate(brain_regions):
            ax = axes[i]
            
            # 各領域の2チャンネルを平均
            obs_region = observed_ecog[:, i*2:(i+1)*2].mean(axis=1)
            pred_region = predicted_ecog[:, i*2:(i+1)*2].mean(axis=1)
            
            # パワースペクトル計算
            freqs_obs, psd_obs = self.compute_power_spectrum(obs_region)
            freqs_pred, psd_pred = self.compute_power_spectrum(pred_region)
            
            # プロット（対数スケール）
            ax.semilogy(freqs_obs, psd_obs, 'k--', label='Observed')
            ax.semilogy(freqs_pred, psd_pred, 'r-', label='Predicted')
            
            ax.set_xlim([0, 100])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD (log scale)')
            ax.set_title(region)
            ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_observed_vs_predicted(self, observed_ecog, predicted_ecog, n_samples=1000):
        """観測と予測のECoG波形を比較表示（20ch）"""
        T = min(len(observed_ecog), len(predicted_ecog), n_samples)
        fig, axes = plt.subplots(10, 2, figsize=(14, 16), sharex=True)
        axes = axes.flatten()
        
        for ch in range(20):
            ax = axes[ch]
            ax.plot(observed_ecog[:T, ch], color='k', linewidth=0.8, label='Observed')
            ax.plot(predicted_ecog[:T, ch], color='r', linewidth=0.8, alpha=0.7, label='Predicted')
            ax.set_title(f'Channel {ch+1}')
            if ch == 0:
                ax.legend(loc='upper right')
        
        axes[-2].set_xlabel('Time (samples)')
        axes[-1].set_xlabel('Time (samples)')
        plt.tight_layout()
        return fig
    
    def analyze_transfer_entropy(self, model, condition='anesthesia'):
        """Transfer Entropyの解析（論文Figure 3）"""
        # この実装は簡略化版
        # 実際は時系列間の情報転送を計算
        
        # z^(2)から各d^(1)への接続強度を可視化
        func_to_local_weights = model.func_to_local.weight.data.cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # ヒートマップ
        sns.heatmap(func_to_local_weights, 
                   xaxis=range(3),  # 3つのz^(2)
                   yaxis=range(15), # 15次元のd^(1)入力
                   cmap='seismic',
                   center=0,
                   cbar_kws={'label': 'Connection Weight'})
        
        ax.set_xlabel('Functional Network (z²)')
        ax.set_ylabel('Local Region Input (d¹)')
        ax.set_title(f'Functional Network Connectivity - {condition}')
        
        return fig
    
    def create_evaluation_report(self, model, test_loader, data_assimilation):
        """総合評価レポートの作成"""
        report = {
            'clustering_metrics': {},
            'prediction_accuracy': {},
            'latent_analysis': {}
        }
        
        # テストデータでの評価
        print("Evaluating model performance...")
        
        with torch.no_grad():
            for batch in test_loader:
                ecog_data = batch['ecog'].to(self.device)
                labels = batch['label'].numpy()
                
                # データ同化でz^(3)を抽出
                z3_trajectory = self.extract_z3_trajectory(
                    data_assimilation, 
                    ecog_data.squeeze(0)
                )
                
                # クラスタリング評価
                if len(np.unique(labels)) > 1:
                    clustering_metrics = self.evaluate_clustering(
                        z3_trajectory, 
                        labels.repeat(len(z3_trajectory))
                    )
                    report['clustering_metrics'].update(clustering_metrics)
                
                break  # 最初のバッチのみ
        
        return report

# 使用例を含むメイン評価スクリプト
def run_evaluation(model, test_loader, config):
    """評価の実行"""
    evaluator = ECoGEvaluator(model)
    data_assimilation = DataAssimilation(model, config)
    
    # 評価レポートの作成
    report = evaluator.create_evaluation_report(
        model, test_loader, data_assimilation
    )
    
    print("Evaluation Report:")
    print(f"Silhouette Score: {report['clustering_metrics'].get('silhouette_score', 'N/A')}")
    print(f"KNN Accuracy: {report['clustering_metrics'].get('knn_accuracy', 'N/A')}")
    
    return report