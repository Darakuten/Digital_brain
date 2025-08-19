# Digital Brainプロジェクトの主要コンポーネント分析

## プロジェクト概要
Digital Brainは**PV-RNN（Predictive-coding Variational Recurrent Neural Network）**を使用したECoG（皮質電位図）データのデジタルツインシミュレーションシステムです。

## 主要コンポーネント

### 1. **ネットワークアーキテクチャ** (`networks/`)
- **`pvrnn.py`**: 多層PV-RNNのメインアーキテクチャ
- **`pvrnn_layer.py`**, **`pvrnn_top_layer.py`**: 各層の実装
- **`integrated_network.py`**: ネットワーク統合機能
- **`output.py`**: 出力層ユーティリティ

### 2. **データ処理** (`Digital Brain/Data/`)
- **`data_loader.py`**: NeuroTychoデータベースからのECoGデータ読み込み
  - 128チャンネル → 20チャンネル（10脳領域×2チャンネル）
  - 前処理: Common Median Reference、外れ値除去、正規化
  - 覚醒/麻酔状態の分離
- **`analyze_ecog_data.py`**: データ分析ユーティリティ

### 3. **モデル実装** (`Digital Brain/model/`)
- **`ecog_vrnn_model.py`**: ECoGデジタルツインの核心実装
  - 3層階層: Global State (τ=100000) → Functional Network (τ=4) → Local Regions (τ=2)
  - 各層でのKLダイバージェンス計算
  - 事前・事後分布の管理
- **`main_ecog_experiment.py`**: メイン実験実行スクリプト
- **`config.py`**: 設定管理

### 4. **データ同化・訓練** (`Digital Brain/model/`)
- **`data_assimilation.py`**: リアルタイム潜在状態推定
- **`training.py`**: モデル訓練プロセス
- **`evaluation.py`**: 性能評価メトリクス

### 5. **実験機能** (`Digital Brain/experiment/`)
- **`virtual_intervention.py`**: 仮想介入実験
  - z³レベル: 薬物投与シミュレーション（全階層・全脳領域に影響）
  - z²レベル: 機能ネットワーク介入
  - z¹レベル: tDCS/TMS刺激シミュレーション（特定脳領域への刺激）
- **`discriminator.py`**: 覚醒/麻酔状態の分類器（3D CNN）

### 6. **実行例** (`Digital Brain/example/`)
- **`example_of_training_rnn_part.py`**: 訓練実行スクリプト
- **`dataset.py`**: データセット管理
- **`utilities.py`**: 共通ユーティリティ
- **`v_rnn.py`**: バリエーショナルRNNの実装
- **`plot_development.py`**: 開発軌跡の可視化
- **`network_config.yaml`**: ハイパーパラメータ設定
- `result_training/`, `trained_model/`, `target/`: 結果・学習済みモデル・ターゲット配置先

## 階層構造
1. **Level 3 (Global State)**: 
   - τ=100000（非常に遅い時定数）
   - z_size=1、d_size=2
   - 全体的な意識状態を制御
   
2. **Level 2 (Functional Network)**: 
   - τ=4（中程度の時定数）
   - z_size=3、d_size=15
   - 機能ネットワークレベルの制御
   
3. **Level 1 (Local Regions)**: 
   - τ=2（速い時定数）
   - z_size=10、d_size=15（各脳領域）
   - 10個の脳領域: FP, DLPFC, PM, M1, S1, IPS, AT, AC, HV, V1

## 脳領域マッピング
- **FP**: Frontal Pole（前頭極）
- **DLPFC**: Dorsolateral Prefrontal Cortex（背外側前頭前皮質）
- **PM**: Premotor Cortex（運動前野）
- **M1**: Primary Motor Cortex（一次運動野）
- **S1**: Primary Somatosensory Cortex（一次体性感覚野）
- **IPS**: Intraparietal Sulcus（頭頂溝）
- **AT**: Anterior Temporal Cortex（前側頭皮質）
- **AC**: Auditory Cortex（聴覚野）
- **HV**: Higher Visual Cortex（高次視覚野）
- **V1**: Primary Visual Cortex（一次視覚野）

## 主要機能

### データ処理パイプライン
1. **生データ読み込み**: NeuroTychoデータベースから128チャンネルECoGデータ
2. **チャンネル選択**: 各脳領域から2チャンネルずつ選択（計20チャンネル）
3. **前処理**: 
   - Common Median Reference適用
   - 外れ値除去（8σ基準、2秒ビン単位）
   - 正規化 [-1000μV, 1000μV] → [-0.8, 0.8]
4. **セグメント化**: 2000タイムステップ（2秒）単位

### モデル特性
- **事前分布**: p(z_t|d_{t-1})、MTRNN状態から計算
- **事後分布**: q(z_t|X)、適応的内部状態aから計算
- **KLダイバージェンス**: 各層でメタプライア重みW^(l)による重み付け
- **出力生成**: 各脳領域の決定論的状態から2チャンネルの出力

### 実験機能
- **リアルタイム状態推定**: 覚醒↔麻酔遷移の検出
- **仮想介入実験**: 
  - 薬物効果のシミュレーション
  - 脳刺激（tDCS/TMS）効果の予測
- **クラスタリング分析**: z³空間での個体差・状態差の分析
- **性能評価**: シルエット幅、k-NN分類精度

## ファイル構造
```
Digital Brain/
├── Data/
│   ├── data_loader.py              # データ読み込み・前処理
│   ├── analyze_ecog_data.py        # データ分析
│   └── 20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
├── model/
│   ├── ecog_vrnn_model.py          # メインモデル
│   ├── main_ecog_experiment.py     # 実験実行
│   ├── training.py                 # 訓練プロセス
│   ├── data_assimilation.py        # データ同化
│   ├── evaluation.py               # 評価
│   └── config.py                   # 設定
├── experiment/
│   ├── virtual_intervention.py     # 仮想介入実験
│   └── discriminator.py            # 分類器
├── networks/
│   ├── pvrnn.py                    # PV-RNNアーキテクチャ
│   ├── pvrnn_layer.py              # PV-RNN層
│   ├── pvrnn_top_layer.py          # 最上位層
│   ├── integrated_network.py       # ネットワーク統合
│   └── output.py                   # 出力ユーティリティ
└── example/
    ├── example_of_training_rnn_part.py  # 訓練例
    ├── dataset.py                       # データセット
    ├── utilities.py                     # ユーティリティ
    ├── v_rnn.py                         # VRNN実装
    ├── plot_development.py              # 可視化
    └── network_config.yaml              # ハイパーパラメータ
```

## 依存関係
- **Python**: 3.8+ 推奨（ColabはOK）
- **PyTorch**: 2.x（CUDA環境推奨）
- **その他**: numpy, scipy, scikit-learn, matplotlib, seaborn, tqdm, PyYAML, requests

インストール例:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU環境の場合
pip install numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml requests
```




## 使用方法

### 1. 訓練（例・合成データ／テキストターゲット）
```bash
cd "Digital Brain/example/"
python example_of_training_rnn_part.py
```

### 2. メイン実験実行（NeuroTycho実データ）
- `model/main_ecog_experiment.py` の `create_ecog_dataloaders` 呼び出しで、戻り値は3つです。
  - コード内を以下のように修正してください（アンパック数の整合性):
  
    ```python
    # 修正前
    train_loader, test_loader = create_ecog_dataloaders(...)
    # 修正後
    train_loader, test_loader, realtime_loader = create_ecog_dataloaders(...)
    ```
- データパスを実データ配置に合わせて設定してください:
  - `data_path='path/to/neurotycho/data'` を NeuroTycho データの実パス（例: `/content/drive/MyDrive/NeuroTycho`）に変更。
- インポートエラー（`ModuleNotFoundError: No module named 'Data'`）が出る場合:
  - 実行前に`Digital Brain`直下で`PYTHONPATH`を追加してから実行してください。
    ```bash
    cd "Digital Brain"
    export PYTHONPATH=$(pwd)
    cd model
    python main_ecog_experiment.py
    ```

実行:
```bash
cd "Digital Brain/model/"
python main_ecog_experiment.py
```

### 3. 仮想介入実験
```bash
cd "Digital Brain/experiment/"
python virtual_intervention.py
```



## Colabでの実行
1. **GPUランタイム有効化**: ランタイム > ランタイムのタイプを変更 > ハードウェア アクセラレータ: GPU
2. **リポジトリ取得**:
   ```bash
   git clone https://github.com/<your-account>/pvrnn_sa-master.git
   cd pvrnn_sa-master
   ```
   もしくは、ZipをColabへアップロードして展開。
3. **依存関係のインストール**:
   ```bash
   pip install numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml requests
   # GPUを使う場合は以下（環境に合わせてバージョン調整）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. **（オプション）Google Driveをマウントしてデータを配置**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # NeuroTychoデータを /content/drive/MyDrive/NeuroTycho/ に配置
   ```
   期待するディレクトリ例:
   ```
   /content/drive/MyDrive/NeuroTycho/
     └── 20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
         ├── Session1/ (ECoG_ch*.mat, Condition.mat)
         ├── Session2/ (ECoG_ch*.mat, Condition.mat)
         └── Session3/ (ECoG_ch*.mat, Condition.mat)
   ```
5. **実行例**:
   - 例1: 合成データの訓練のみ（外部データ不要）
     ```bash
     cd "/content/pvrnn_sa-master/Digital Brain/example"
     python example_of_training_rnn_part.py
     ```
   - 例2: 実データでメイン実験
     ```bash
     cd "/content/pvrnn_sa-master/Digital Brain/model"
     # ファイルを編集: main_ecog_experiment.py の data_path を
     # data_path='/content/drive/MyDrive/NeuroTycho' に変更
     # かつ create_ecog_dataloaders の戻り値を3つ受け取るように修正
     # さらにインポートエラーが出る場合は:
     export PYTHONPATH="/content/pvrnn_sa-master/Digital Brain"
     python main_ecog_experiment.py
     ```

補足:
- Colabのパスには空白が含まれるため、`cd "Digital Brain/..."` のように引用符で囲ってください。
- 大規模学習（`n_epochs=200000`）は時間とコストが大きいため、検証用には小さなエポック数に調整してください（`model/config.py`）。
