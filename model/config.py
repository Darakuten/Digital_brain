# config.py
config = {
    # モデルアーキテクチャ
    # global_d_size は実装で未参照のため削除（ECoGDigitalTwinRevised は内部で d_size=2 を使用）
    'global_z_size': 2,
    'global_tau': 10000, #変更可能
    
    'func_d_size': 15,
    'func_z_size': 3,
    'func_tau': 4,
    
    'local_d_size': 15,   # 論文の運用上はレベル単位の決定論的次元として15で妥当（zごとに15ではない）
    'local_z_size': 10,
    'local_tau': 2,
    
    # メタプライア W^(l)（論文では全レベル一様 0.001）
    'W_global': 0.001,
    'W_global_t0': 0.001,  # 区別しない（実質同値運用）
    'W_func': 0.001,
    'W_local': 0.001,
    
    # 学習パラメータ
    'learning_rate': 0.001,
    'batch_size': 16,
    'n_epochs': 200,#200000
    
    # データ同化パラメータ（論文準拠）
    'window_size': 500,
    'n_updates': 100,
    'n_updates_initial': 100,    # 論文に直接の言及はないが初期安定化のため維持
    'da_learning_rate': 0.001,   # 論文に合わせて 0.001 に修正
    'da_stride': 10           # 新規追加: データ同化のスライド幅。論文準拠は1
}
