# main_ecog_experiment.py
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime # datetimeモジュールを追加

# カスタムモジュールのインポート
from Data.data_loader import NeuroTychoECoGDataset, create_ecog_dataloaders
from ecog_vrnn_model import ECoGDigitalTwinRevised
from training import ECoGTrainerRevised
from data_assimilation import DataAssimilationRevised
from evaluation import ECoGEvaluator


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_history(history, out_png):
    plt.figure(figsize=(8, 5))
    if 'loss' in history:
        plt.plot(history['loss'], label='Free Energy (loss)')
    if 'recon_error' in history:
        plt.plot(history['recon_error'], label='Reconstruction Error')
    if 'kl_loss' in history:
        plt.plot(history['kl_loss'], label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    # 設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 実行ごとの結果ディレクトリを作成
    base_results_dir = os.path.join(os.path.dirname(__file__), 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = ensure_dir(os.path.join(base_results_dir, timestamp))
    print(f"Results will be saved to: {results_dir}")
    
    # 設定の読み込み
    from config import config
    
    # データローダーの作成
    print("Creating data loaders...")
    train_loader, test_loader, realtime_loader = create_ecog_dataloaders(
        data_path='/Users/mz/Downloads/pvrnn_sa-master/Digital Brain/Data',
        batch_size=config['batch_size'],
        train_subjects=[1, 2, 3],
        test_subject=4,
        test_protocol='z3',
        random_state=42
    )

    # データ形式のメタ情報出力
    def numpy_int(x):
        return int(x) if hasattr(x, 'item') else int(x)

    train_ds = train_loader.dataset
    test_ds = test_loader.dataset
    rt_ds = realtime_loader.dataset

    # 形状の取得（train/testはsegmentsから、realtimeはサンプルから）
    train_num = len(train_ds)
    test_num = len(test_ds)
    rt_num = len(rt_ds)

    # train_ds.segments.shape[1] や rt_sample.shape[0] は Dataset に依存するため、より堅牢にアクセス
    train_seq_len = int(train_ds.segment_length) if hasattr(train_ds, 'segment_length') else (int(train_ds.segments.shape[1]) if hasattr(train_ds, 'segments') and train_ds.segments.shape[0] > 0 else 0)
    train_n_ch = int(train_ds.segments.shape[2]) if hasattr(train_ds, 'segments') and train_ds.segments.shape[0] > 0 else 0

    test_seq_len = int(test_ds.segment_length) if hasattr(test_ds, 'segment_length') else (int(test_ds.segments.shape[1]) if hasattr(test_ds, 'segments') and test_ds.segments.shape[0] > 0 else 0)
    test_n_ch = int(test_ds.segments.shape[2]) if hasattr(test_ds, 'segments') and test_ds.segments.shape[0] > 0 else 0

    rt_sample = rt_ds[0]['ecog'] if rt_num > 0 else None
    rt_seq_len = int(rt_sample.shape[0]) if rt_sample is not None else 0
    rt_n_ch = int(rt_sample.shape[1]) if rt_sample is not None else 0

    dataset_info = {
        'device': str(device),
        'config': {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in config.items()},
        'protocol': {
            'test_protocol': 'z3',
            'realtime_sequences': 50,
            'z3_sequences_per_condition': 50,
            'z2z1_sequences_per_condition': 25,
            'segment_length': 2000
        },
        'train': {
            'num_sequences': int(train_num),
            'seq_len': int(train_seq_len),
            'n_channels': int(train_n_ch),
            'label_counts': {
                'anesthesia(0)': int(np.sum(train_ds.labels == 0)) if hasattr(train_ds, 'labels') else None,
                'awake(1)': int(np.sum(train_ds.labels == 1)) if hasattr(train_ds, 'labels') else None,
            },
            'subjects': sorted(list(set(map(lambda x: str(x), train_ds.subject_info.tolist())))) if hasattr(train_ds, 'subject_info') else None,
        },
        'test': {
            'num_sequences': int(test_num),
            'seq_len': int(test_seq_len),
            'n_channels': int(test_n_ch),
            'label_counts': {
                'anesthesia(0)': int(np.sum(test_ds.labels == 0)) if hasattr(test_ds, 'labels') else None,
                'awake(1)': int(np.sum(test_ds.labels == 1)) if hasattr(test_ds, 'labels') else None,
            },
            'subjects': sorted(list(set(map(lambda x: str(x), test_ds.subject_info.tolist())))) if hasattr(test_ds, 'subject_info') else None,
        },
        'realtime': {
            'num_sequences': int(rt_num),
            'seq_len': int(rt_seq_len),
            'n_channels': int(rt_n_ch),
            'note': 'First 25 are Awake→Anesthesia, next 25 are Anesthesia→Awake' if rt_num else ''
        }
    }
    save_json(dataset_info, os.path.join(results_dir, 'dataset_info.json'))

    # モデルの初期化
    print("Initializing model...")
    model = ECoGDigitalTwinRevised(config, device=device)
    
    # 訓練
    print("Starting training...")
    trainer = ECoGTrainerRevised(model, config, device)

    # CSVヘッダを先に出力
    history_csv = os.path.join(results_dir, 'train_history.csv')
    with open(history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'free_energy', 'reconstruction_error', 'kl_loss'])

    stop_flag_path = os.path.join(results_dir, 'STOP')

    try:
        for epoch in range(config['n_epochs']):
            # STOP ファイル検知
            if os.path.exists(stop_flag_path):
                print(f"STOP file detected at {stop_flag_path}. Saving and exiting...")
                break

            train_stats = trainer.train_epoch(train_loader, epoch)
            
            print(f"Epoch {epoch}: Loss={train_stats['loss']:.4f}, "
                  f"Recon={train_stats['recon_error']:.4f}, "
                  f"KL={train_stats['kl_loss']:.4f}")
            
            # 履歴をCSVに逐次追記（確認しやすく）
            with open(history_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_stats['loss'], train_stats['recon_error'], train_stats['kl_loss']])

            # 100エポックごとにJSONとプロットを更新
            if epoch % 100 == 0:
                save_json(trainer.train_history, os.path.join(results_dir, 'train_history.json'))
                plot_history(trainer.train_history, os.path.join(results_dir, 'loss_curve.png'))
            
            # チェックポイント保存
            if epoch % 5000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'train_history': trainer.train_history,
                    'config': config
                }, os.path.join(results_dir, f'checkpoint_epoch_{epoch}.pth'))
    except KeyboardInterrupt:
        # Ctrl-C 安全終了: 直近チェックポイントと履歴を保存
        print("KeyboardInterrupt detected. Saving last state and exiting...")
        save_json(trainer.train_history, os.path.join(results_dir, 'train_history.json'))
        plot_history(trainer.train_history, os.path.join(results_dir, 'loss_curve.png'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_history': trainer.train_history,
            'config': config
        }, os.path.join(results_dir, f'checkpoint_interrupt_epoch_{epoch}.pth'))
    
    # 最終履歴の保存
    save_json(trainer.train_history, os.path.join(results_dir, 'train_history.json'))
    plot_history(trainer.train_history, os.path.join(results_dir, 'loss_curve.png'))

    # データ同化のテスト
    print("\nTesting data assimilation...")
    da = DataAssimilationRevised(model, config, device)
    
    # テストシーケンスの作成（覚醒↔麻酔の遷移：RealtimeTestDatasetの50本のうち先頭1本を使用）
    test_data = next(iter(realtime_loader))
    test_sequence = test_data['ecog'][0].numpy()  # (seq_len, 20)
    
    # データ同化実行
    z3_trajectory, predicted_ecog = da.assimilate(test_sequence)
    
    # 評価
    print("\nEvaluating results...")
    evaluator = ECoGEvaluator(model, device)
    
    # z^3の軌跡プロット
    fig = evaluator.plot_z3_trajectory(z3_trajectory, transition_point=2500)
    fig.savefig(os.path.join(results_dir, 'z3_trajectory.png'))
    # 観測 vs 予測 ECoG の比較
    fig_cmp = evaluator.plot_observed_vs_predicted(test_sequence, predicted_ecog, n_samples=1000)
    fig_cmp.savefig(os.path.join(results_dir, 'observed_vs_predicted.png'))
    
    # 追加: 全50シーケンスを一括処理
    traj_dir = ensure_dir(os.path.join(results_dir, 'realtime_z3_trajs'))
    print("\nProcessing all realtime sequences (50 total)...")
    for seq_idx, batch in enumerate(realtime_loader):
        try:
            ecog_seq = batch['ecog'][0].numpy()
            # RealtimeTestDataset の __getitem__ が is_transition を返すことを前提
            is_awake_to_anesthesia = batch.get('is_transition', torch.tensor([True])).item()
            transition_point = 2500 if is_awake_to_anesthesia else 1500
            
            z3_traj, pred_seq = da.assimilate(ecog_seq)
            np.save(os.path.join(traj_dir, f'seq_{seq_idx+1:04d}.npy'), z3_traj)
            np.save(os.path.join(traj_dir, f'seq_{seq_idx+1:04d}_pred.npy'), pred_seq)
            fig_i = evaluator.plot_z3_trajectory(z3_traj, transition_point=transition_point)
            fig_i.savefig(os.path.join(traj_dir, f'seq_{seq_idx+1:04d}.png'))
            fig_cmp_i = evaluator.plot_observed_vs_predicted(ecog_seq, pred_seq, n_samples=1000)
            fig_cmp_i.savefig(os.path.join(traj_dir, f'seq_{seq_idx+1:04d}_obs_vs_pred.png'))
        except Exception as e:
            print(f"Sequence {seq_idx+1}: error {e}")
    
    print("Experiment completed!")

if __name__ == "__main__":
    main()