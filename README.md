# pwscup2024-hal

[CSS2024](https://www.iwsec.org/css/2024/) の併設ワークショップで開催された [PWSCup2024](https://www.iwsec.org/pws/2024/cup24.html) に参加した チーム12 HAL の公開リポジトリです.

本リポジトリには, 予備戦・本戦の匿名化フェーズ, 攻撃フェーズで使用したコードが含まれています.

## ディレクトリ構成
ディレクトリ構成は以下の通りです.
```
├── anonymize/ :匿名化フェーズ
│   ├── anonymize_main.py :本戦の攻撃プログラム
│   ├── anonymize_pre.py :予備戦の攻撃プログラム
│   └── calc_cramersV.ipynb :クラメールの連関係数を計算するプログラム
└── attack/ :攻撃フェーズ
    ├── modeSimpleAttack.py :シンプルな最頻値攻撃プログラム
    ├── sampleAttack_kai.py :サンプル攻撃プログラムの改良版
    └── subsetHammingAttack.py :サブセットごとの攻撃プログラム
```
