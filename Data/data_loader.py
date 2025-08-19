# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os
from scipy.signal import medfilt
from scipy import signal
import glob

# === 追加: 被験者→フォルダ/セッション設定の解決 ===
# - 明示設定があればそれを優先
# - Condition.mat のラベルに基づく抽出を基本とし、セッション制約があれば適用
SUBJECT_CONFIG = {
    # 例: 既存の20120814MD（論文設定に合わせた想定）
    1: {
        'folder': '20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128',
        # セッション制約が不要なら省略可
        'sessions': {
            'awake': None,       # Noneは全セッションからConditionに従って抽出
            'anesthesia': None,
        }
    },
    # ユーザ指定: 20110525KTMD は Session3 が Anesthesia, Session4 まで存在
    4: {
        'folder': '20110525KTMD_Anesthesia+and+Sleep_Kin2_Toru+Yanagawa_mat_ECoG128',
        'sessions': {
            'awake': None,          # AwakeはConditionラベルから抽出（制限なし）
            'anesthesia': {3},      # AnesthesiaはSession3のみを使用
        }
    },
    # 他の被験者(2,3)はData配下のフォルダ順で自動対応（必要ならここに追記）
}


def _discover_subject_folders(data_path):
    """Data配下の候補フォルダを列挙する。"""
    candidates = []
    for name in os.listdir(data_path):
        full = os.path.join(data_path, name)
        if os.path.isdir(full) and name.endswith('ECoG128'):
            candidates.append(name)
    return sorted(candidates)


def _resolve_subject_folder_and_rule(data_path, subject_id):
    """subject_idから対象フォルダとセッション制約を解決する。
    - intで既知: SUBJECT_CONFIGを優先
    - 文字列: そのままフォルダ名とみなす
    - 未知のint: Data配下の候補を昇順に並べ、1始まりのインデックスで対応
    戻り値: (folder_name, sessions_rule_dict or None)
    """
    # 文字列ならフォルダ名指定とみなす
    if isinstance(subject_id, str):
        folder = subject_id
        return folder, None

    # 既知設定
    if subject_id in SUBJECT_CONFIG:
        conf = SUBJECT_CONFIG[subject_id]
        return conf['folder'], conf.get('sessions', None)

    # 自動割当
    candidates = _discover_subject_folders(data_path)
    if 1 <= subject_id <= len(candidates):
        return candidates[subject_id - 1], None

    raise ValueError(f"Cannot resolve folder for subject_id={subject_id}. Please add to SUBJECT_CONFIG or place data correctly.")


class NeuroTychoECoGDataset(Dataset):
    """
    Neurotychodatabaseからの ECoGデータセット
    論文: 128チャンネル → 20チャンネル（10脳領域 × 2チャンネル）
    """
    
    # 脳領域の定義（論文より）
    BRAIN_REGIONS = {
        'FP': 'Frontal Pole',
        'DLPFC': 'Dorsolateral Prefrontal Cortex', 
        'PM': 'Premotor Cortex',
        'M1': 'Primary Motor Cortex',
        'S1': 'Primary Somatosensory Cortex',
        'IPS': 'Intraparietal Sulcus',
        'AT': 'Anterior Temporal Cortex',
        'AC': 'Auditory Cortex',
        'HV': 'Higher Visual Cortex',
        'V1': 'Primary Visual Cortex'
    }
    
    def __init__(self, data_path, subject_ids, condition='all', segment_length=2000, 
                 sampling_rate=1000, normalize_range=(-0.8, 0.8), is_training=True,
                 max_sequences_per_condition=None, random_state=None):
        """
        Args:
            data_path: データファイルのパス（Data直下）
            subject_ids: 対象個体のIDリスト（int or フォルダ名str）
            condition: 'anesthesia', 'awake', or 'all'
            segment_length: セグメント長（タイムステップ数）= 2000
            sampling_rate: サンプリングレート = 1000Hz
            normalize_range: 正規化範囲 = (-0.8, 0.8)
            is_training: 訓練用データかどうか
            max_sequences_per_condition: 非学習時に各条件からサンプリングする最大本数（例: z³=50, z²/z¹=25）
            random_state: 乱数シード
        """
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.normalize_range = normalize_range
        self.condition = condition
        self.subject_ids = subject_ids
        self.is_training = is_training
        self.max_sequences_per_condition = max_sequences_per_condition
        self.random_state = random_state
        
        # データの読み込みと前処理
        self.segments, self.labels, self.subject_info = self._load_and_preprocess(data_path)
        
    def _load_and_preprocess(self, data_path):
        """データの読み込みと前処理"""
        all_segments = []
        all_labels = []
        all_subject_info = []
        
        rng = np.random.default_rng(self.random_state)
        
        for subject_id in self.subject_ids:
            # 1. 生データの読み込み（フォルダ解決 + セッション自動検出 + Condition優先）
            data_anesthesia, data_awake = self._load_raw_data(data_path, subject_id)
            
            # 各条件について処理
            for condition, raw_data in [('anesthesia', data_anesthesia), 
                                       ('awake', data_awake)]:
                if raw_data is None or raw_data.size == 0:
                    continue
                    
                # 2. チャンネル選択（128ch → 20ch）
                selected_channels = self._select_channels(raw_data)
                
                # 3. Common Median Reference
                referenced_data = self._apply_common_median_reference(selected_channels)
                
                # 4. 外れ値除去（ビン単位）
                cleaned_data, outlier_percentage = self._remove_outliers_binwise(referenced_data)
                print(f"Subject {subject_id} - {condition}: {outlier_percentage:.1f}% data removed")
                
                # 5. 正規化 [-1000μV, 1000μV] → [-0.8, 0.8]
                normalized_data = self._normalize(cleaned_data)
                
                # 6. セグメント化（2000タイムステップ）
                segments, labels = self._segment_data(normalized_data, condition)
                
                # 訓練用データ: 各条件から固定本数（例:12）をランダム選択
                if self.is_training and self.condition in ['all', condition]:
                    n_sequences = 12
                    if len(segments) >= n_sequences:
                        indices = rng.choice(len(segments), n_sequences, replace=False)
                        segments = segments[indices]
                        labels = labels[indices]
                
                # 評価用データ: 論文プロトコルに沿って各条件から最大本数を抽出
                if (not self.is_training) and (self.max_sequences_per_condition is not None) \
                   and self.condition in ['all', condition]:
                    n_sequences = int(self.max_sequences_per_condition)
                    if len(segments) >= n_sequences:
                        indices = rng.choice(len(segments), n_sequences, replace=False)
                        segments = segments[indices]
                        labels = labels[indices]
                
                all_segments.extend(segments)
                all_labels.extend(labels)
                all_subject_info.extend([subject_id] * len(segments))
        
        return np.array(all_segments), np.array(all_labels), np.array(all_subject_info)
    
    def _load_raw_data(self, data_path, subject_id):
        """生データの読み込み。フォルダ名とセッション制約を解決し、Conditionに基づいて抽出。"""
        import scipy.io
        
        folder_name, sessions_rule = _resolve_subject_folder_and_rule(data_path, subject_id)
        base_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Data folder not found: {base_path}")
        
        # 利用可能なSessionを自動検出
        session_dirs = sorted(glob.glob(os.path.join(base_path, 'Session*')))
        sessions_data = {}
        conditions_data = {}
        available_session_nums = []
        
        for session_path in session_dirs:
            try:
                session_num = int(os.path.basename(session_path).replace('Session', ''))
            except Exception:
                continue
            available_session_nums.append(session_num)
            
            # Condition読み込み
            condition_file = os.path.join(session_path, 'Condition.mat')
            if os.path.exists(condition_file):
                condition_mat = scipy.io.loadmat(condition_file)
                conditions_data[session_num] = {
                    'time': condition_mat.get('ConditionTime', np.array([[]]))[0] if 'ConditionTime' in condition_mat else np.array([[]]),
                    'labels': condition_mat.get('ConditionLabel', np.array([[]]))[0] if 'ConditionLabel' in condition_mat else np.array([[]])
                }
            
            # ECoGチャンネルデータの読み込み
            ecog_files = glob.glob(os.path.join(session_path, 'ECoG_ch*.mat'))
            if ecog_files:
                session_channels = {}
                for file_path in sorted(ecog_files):
                    try:
                        ch_num = int(file_path.split('_ch')[1].split('.')[0])
                    except Exception:
                        continue
                    mat_data = scipy.io.loadmat(file_path)
                    ecog_key = f'ECoGData_ch{ch_num}'
                    if ecog_key in mat_data:
                        session_channels[ch_num] = mat_data[ecog_key].flatten()
                if session_channels:
                    sessions_data[session_num] = session_channels
        
        # 条件ごとに抽出（Conditionラベルがあればそれを用い、セッション制約があれば適用）
        data_anesthesia_list = []
        data_awake_list = []
        
        # セッション制約の取り出し
        allowed_awake = None
        allowed_anesthesia = None
        if sessions_rule is not None:
            allowed_awake = sessions_rule.get('awake', None)
            allowed_anesthesia = sessions_rule.get('anesthesia', None)
        
        for session_num, session_channels in sessions_data.items():
            cond_info = conditions_data.get(session_num, None)
            
            # Awake抽出
            if allowed_awake is None or session_num in allowed_awake:
                awake_data = self._extract_condition_data(
                    session_channels, cond_info,
                    ['AwakeEyesOpened', 'AwakeEyesClosed']
                ) if cond_info is not None else None
                if awake_data is not None and awake_data.size > 0:
                    data_awake_list.append(awake_data)
            
            # Anesthesia抽出
            if allowed_anesthesia is None or session_num in allowed_anesthesia:
                anesthesia_data = self._extract_condition_data(
                    session_channels, cond_info,
                    ['Anesthetized']
                ) if cond_info is not None else None
                if anesthesia_data is not None and anesthesia_data.size > 0:
                    data_anesthesia_list.append(anesthesia_data)
        
        data_anesthesia = np.concatenate(data_anesthesia_list, axis=1) if len(data_anesthesia_list) else None
        data_awake = np.concatenate(data_awake_list, axis=1) if len(data_awake_list) else None
        
        return data_anesthesia, data_awake
    
    def _extract_condition_data(self, session_channels, condition_info, target_conditions):
        """指定された条件のデータを抽出"""
        if not session_channels or condition_info is None:
            return None
            
        # 条件のタイムスタンプを取得
        condition_times = condition_info['time']
        condition_labels = condition_info['labels']
        
        # 対象条件の時間範囲を特定
        time_ranges = []
        start_time = None
        for i, label in enumerate(condition_labels):
            label_str = label[0] if isinstance(label, np.ndarray) else str(label)
            for target_condition in target_conditions:
                if target_condition in label_str:
                    if 'Start' in label_str:
                        start_time = condition_times[i]
                    elif 'End' in label_str and start_time is not None:
                        end_time = condition_times[i]
                        time_ranges.append((int(start_time * 1000), int(end_time * 1000)))
                        start_time = None
        
        if not time_ranges:
            return None
        
        # 全チャンネルのデータを結合
        n_channels = len(session_channels)
        combined_data = []
        
        for start_sample, end_sample in time_ranges:
            segment_data = []
            for ch_num in sorted(session_channels.keys()):
                ch_data = session_channels[ch_num]
                if start_sample < len(ch_data) and end_sample <= len(ch_data):
                    segment_data.append(ch_data[start_sample:end_sample])
            if segment_data:
                combined_data.append(np.array(segment_data))
        
        if combined_data:
            return np.concatenate(combined_data, axis=1)
        
        return None
    
    def _select_channels(self, raw_data):
        """
        チャンネル選択（各脳領域から2チャンネルずつ）
        実際の実装では、電極配置図に基づいて各脳領域に対応する
        チャンネルのマッピングが必要
        """
        # 仮のチャンネルマッピング（実際は電極配置に基づく）
        channel_mapping = {
            'FP': [0, 1],      # 前頭極
            'V1': [10, 11],    # 一次視覚野
            'HV': [20, 21],    # 高次視覚野
            'AC': [30, 31],    # 聴覚野
            'S1': [40, 41],    # 一次体性感覚野
            'M1': [50, 51],    # 一次運動野
            'IPS': [60, 61],   # 頭頂溝
            'PM': [70, 71],    # 運動前野
            'DLPFC': [80, 81], # 背外側前頭前皮質
            'AT': [90, 91]     # 前側頭皮質
        }
        
        selected_indices = []
        for region, channels in channel_mapping.items():
            selected_indices.extend(channels)
        
        return raw_data[selected_indices, :]
    
    def _apply_common_median_reference(self, data):
        """Common Median Reference の適用"""
        median_signal = np.median(data, axis=0)
        return data - median_signal[np.newaxis, :]
    
    def _remove_outliers_binwise(self, data, threshold=8):
        """
        外れ値除去（2秒ビン単位で8σを超えるビンとその隣接ビンを除外）
        """
        bin_size = 2 * self.sampling_rate  # 2秒 = 2000タイムステップ
        n_bins = data.shape[1] // bin_size
        
        # ビンごとの有効性を記録
        valid_bins = np.ones(n_bins, dtype=bool)
        
        for bin_idx in range(n_bins):
            start = bin_idx * bin_size
            end = min((bin_idx + 1) * bin_size, data.shape[1])
            bin_data = data[:, start:end]
            
            # ビン内の標準偏差と平均
            bin_std = np.std(bin_data)
            bin_mean = np.mean(bin_data)
            
            # 8σを超える値が存在するか確認
            if np.any(np.abs(bin_data - bin_mean) > threshold * bin_std):
                valid_bins[bin_idx] = False
                # 隣接ビンも無効化
                if bin_idx > 0:
                    valid_bins[bin_idx - 1] = False
                if bin_idx < n_bins - 1:
                    valid_bins[bin_idx + 1] = False
        
        # 有効なビンのデータのみを保持
        valid_data = []
        for bin_idx in range(n_bins):
            if valid_bins[bin_idx]:
                start = bin_idx * bin_size
                end = min((bin_idx + 1) * bin_size, data.shape[1])
                valid_data.append(data[:, start:end])
        
        if len(valid_data) > 0:
            cleaned_data = np.concatenate(valid_data, axis=1)
        else:
            cleaned_data = np.array([]).reshape(data.shape[0], 0)
        
        # 除外されたデータの割合
        outlier_percentage = (1 - np.sum(valid_bins) / n_bins) * 100
        
        return cleaned_data, outlier_percentage
    
    def _normalize(self, data):
        """正規化 [-1000μV, 1000μV] → [-0.8, 0.8]"""
        # クリッピング
        data = np.clip(data, -1000, 1000)
        # 線形変換
        normalized = data * 0.8 / 1000
        return normalized
    
    def _segment_data(self, data, condition):
        """データのセグメント化"""
        segments = []
        labels = []
        
        # セグメント数の計算
        n_segments = data.shape[1] // self.segment_length
        
        for i in range(n_segments):
            start = i * self.segment_length
            end = start + self.segment_length
            
            if end > data.shape[1]:
                break
                
            segment = data[:, start:end].T  # (time, channels)
            segments.append(segment)
            
            # ラベル付け
            if condition == 'anesthesia':
                labels.append(0)
            else:  # awake
                labels.append(1)
        
        return np.array(segments), np.array(labels)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return {
            'ecog': torch.FloatTensor(self.segments[idx]),
            'label': torch.LongTensor([self.labels[idx]]),
            'condition': 'anesthesia' if self.labels[idx] == 0 else 'awake',
            'subject': self.subject_info[idx]
        }


class RealtimeTestDataset(Dataset):
    """リアルタイム潜在状態推定実験用のデータセット"""
    
    def __init__(self, data_path, test_subject_id, n_sequences=50, 
                 awake_length=2500, anesthesia_length=1500, taper_width=30):
        """
        Args:
            data_path: データファイルのパス
            test_subject_id: テスト個体のID
            n_sequences: 生成するシーケンス数（各順序で25個ずつ）
            awake_length: 覚醒状態の長さ
            anesthesia_length: 麻酔状態の長さ
            taper_width: クロスフェードのテーパリング幅（ms）
        """
        self.taper_width = taper_width
        self.sequences = self._create_sequences(
            data_path, test_subject_id, n_sequences, 
            awake_length, anesthesia_length
        )
    
    def _create_sequences(self, data_path, subject_id, n_sequences, 
                         awake_length, anesthesia_length):
        """状態遷移を含むシーケンスの作成"""
        # 基本データセットの作成
        base_dataset = NeuroTychoECoGDataset(
            data_path, [subject_id], condition='all', 
            is_training=False
        )
        
        # 麻酔下と覚醒下のセグメントを分離
        anesthesia_segments = []
        awake_segments = []
        
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item['label'][0] == 0:
                anesthesia_segments.append(item['ecog'])
            else:
                awake_segments.append(item['ecog'])
        
        sequences = []
        
        # Awake → Anesthesia の順序で25シーケンス
        for _ in range(n_sequences // 2):
            # ランダムにセグメントを選択
            awake_idx = np.random.randint(len(awake_segments))
            anesthesia_idx = np.random.randint(len(anesthesia_segments))
            
            awake_data = awake_segments[awake_idx][:awake_length]
            anesthesia_data = anesthesia_segments[anesthesia_idx][:anesthesia_length]
            
            # クロスフェード適用
            combined = self._apply_crossfade(awake_data, anesthesia_data)
            sequences.append(combined)
        
        # Anesthesia → Awake の順序で25シーケンス
        for _ in range(n_sequences // 2):
            awake_idx = np.random.randint(len(awake_segments))
            anesthesia_idx = np.random.randint(len(anesthesia_segments))
            
            anesthesia_data = anesthesia_segments[anesthesia_idx][:awake_length]
            awake_data = awake_segments[awake_idx][:anesthesia_length]
            
            # クロスフェード適用
            combined = self._apply_crossfade(anesthesia_data, awake_data)
            sequences.append(combined)
        
        return sequences
    
    def _apply_crossfade(self, data1, data2):
        """線形クロスフェードの適用"""
        # テーパリング幅（サンプル数）
        taper_samples = self.taper_width
        
        # フェード係数の作成
        fade_out = np.linspace(1, 0, taper_samples)
        fade_in = np.linspace(0, 1, taper_samples)
        
        # データの結合
        combined = np.zeros((data1.shape[0] + data2.shape[0] - taper_samples, 
                           data1.shape[1]))
        
        # 最初の部分
        combined[:data1.shape[0] - taper_samples] = data1[:-taper_samples]
        
        # クロスフェード部分
        start_idx = data1.shape[0] - taper_samples
        for i in range(taper_samples):
            combined[start_idx + i] = (
                data1[data1.shape[0] - taper_samples + i] * fade_out[i] +
                data2[i] * fade_in[i]
            )
        
        # 最後の部分
        combined[data1.shape[0]:] = data2[taper_samples:]
        
        return torch.FloatTensor(combined)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'ecog': self.sequences[idx],
            'is_transition': idx < 25  # 最初の25個がAwake→Anesthesia
        }


def create_ecog_dataloaders(data_path, batch_size=48, train_subjects=[1, 2, 3], 
                           test_subject=4, test_protocol='z3', random_state=None):
    """
    データローダーの作成（4-fold cross-validation用）
    
    Args:
        data_path: データファイルのパス
        batch_size: バッチサイズ（デフォルト48）
        train_subjects: 訓練用個体のIDリスト（3個体）
        test_subject: テスト用個体のID（1個体）
        test_protocol: 'z3'（100=各50）, 'z2z1'（50=各25）, 'full'（制限なし）
        random_state: 乱数シード
    """
    # プロトコルに基づく最大本数/条件
    max_per_condition = None
    if test_protocol == 'z3':
        max_per_condition = 50
    elif test_protocol in ['z2z1', 'z2', 'z1']:
        max_per_condition = 25
    elif test_protocol in ['full', None]:
        max_per_condition = None
    else:
        max_per_condition = None
    
    # 訓練用データセット
    train_dataset = NeuroTychoECoGDataset(
        data_path, train_subjects, condition='all', is_training=True,
        segment_length=2000, random_state=random_state
    )
    
    # テスト用データセット（論文プロトコル準拠の本数に制限）
    test_dataset = NeuroTychoECoGDataset(
        data_path, [test_subject], condition='all', is_training=False,
        segment_length=2000, max_sequences_per_condition=max_per_condition,
        random_state=random_state
    )
    
    # リアルタイム推定用データセット（固定で50本の遷移シーケンス）
    realtime_dataset = RealtimeTestDataset(data_path, test_subject)
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    realtime_loader = DataLoader(realtime_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, realtime_loader