"""

サンプル攻撃プログラムの改良版. サブセットの結合を最頻値で行う.

"""

import sys
import os
import argparse
from contextlib import redirect_stdout

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore', FutureWarning)

# ハミング距離を計算する関数
def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

# 指定された列のみを読み込む関数
def read_columns(file, cols):
    df = pd.read_csv(file)
    return df[cols]

# 最頻値を計算する関数
def mode(series):
    return series.value_counts().idxmax()

# `*`の場所をTの対応する値で置き換える関数
def fill_placeholders(row, T_row, columns):
    row_filled = row.copy()
    for col in columns:
        if row[col] == '*':
            return T_row[col]

# メインの処理
def main(a_prefix, b_prefix, c_prefix):
    # ファイルの読み込み
    a_file = f'{a_prefix}.csv'
    b_file = f'{b_prefix}.csv'
    a = pd.read_csv(a_file)
    b = pd.read_csv(b_file)

    # a.csvのname列を削除
    if 'Name' in a.columns:
        a = a.drop(columns=['Name'])

    # 列の設定
    c_columns = [
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '653', '1525', '2105', '2193', '2253', '2628', '2872', '3438', '3439', '3440', '3877', '3889'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '260', '653', '673', '1009', '1073', '1525', '1750', '1881', '1967', '2043', '2093', '2105', '2143', '2193', '2399', '2628', '2968', '3479', '3489', '3877', '3889'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '673', '1881', '1920', '2087', '2138'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '673', '810', '885', '1009', '1073', '1097', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017', '2043', '2087', '2093', '2138', '2399', '3438', '3439', '3440'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '673', '810', '1073', '1126', '1702', '2100', '2174', '2253', '2797', '3393', '3466'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '885', '1097', '1654', '2086', '2138', '2872'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '2100', '2143', '2872', '3479'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '1097', '1750', '2021', '2093', '2105', '2628', '2968'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '1920', '2017', '2087'],
        ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '1097', '2628', '2174', '2797', '1073', '2100', '2968', '2105', '2193']
    ]

    final_columns = ['Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '247', '260', '653', '673', '810', '885',
                     '1009', '1073', '1097', '1126', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017',
                     '2021', '2043', '2086', '2087', '2093', '2100', '2105', '2138', '2143', '2174', '2193', '2253',
                     '2399', '2628', '2797', '2872', '2968', '3393', '3438', '3439', '3440', '3466', '3479', '3489',
                     '3877', '3889']

    # 最頻値を選択
    for column in tqdm(final_columns):
        init_flag = True
        for i in range(10):
            if column in c_columns[i]:
                c_file = f'{c_prefix}_{i}.csv'
                if not os.path.exists(c_file):
                    print(f'{c_file} does not exist')
                    sys.exit(1)
                df = pd.read_csv(c_file)
                # df = df.astype('int', errors='ignore') # For team 20
                if init_flag:
                    df_one = pd.DataFrame({str(i): df[column]})
                    init_flag = False
                else:
                    df_one[str(i)] = df[column]
        s_mode = df_one.apply(mode, axis=1)
        if column == final_columns[0]:
            T = pd.DataFrame({column: s_mode})
        else:
            T[column] = s_mode

    # Tの列を指定された順に並べ替え
    T = T[final_columns]

    # 比較のためa,b,Tを文字列型に変換
    a = a.astype(str)
    b = b.astype(str)
    T = T.astype(str)

    # rowとdfを受け取り、rowに対するdfのhamming_distanceの最小値と最小値を取るindexを返す関数
    def return_min_hamdist_idx(row, df):
        ham_dist = df.apply(lambda df_row: hamming_distance(row, df_row), axis = 1)
        min_dist, min_index = ham_dist.min(), ham_dist.idxmin()

        return pd.Series([min_dist, min_index], index=['Dist', 'idx'])

    # 最小のハミング距離を持つ a.csv の行番号を格納するリスト
    min_distances = []
    min_indices = []
    filled_values = []

    # b.csv の各行について、最小のハミング距離を持つ a.csv の行番号を計算
    for b_index, b_row in b.iterrows():
        min_distance = float('inf')
        min_index = -1

        # A(基本属性のdf)に対してrating部を全てb_rowの値で代入する(apply用)
        tmp_a = a.copy()
        tmp_a[b_row.index] = b_row

        # tmp_aの各行に対して、Tを受け取りhamming_distanceの最小値と最小値を取るindexを返す
        a_dist_df = tmp_a.apply(lambda row: return_min_hamdist_idx(row, T), axis = 1)

        min_index = a_dist_df["Dist"].idxmin()
        T_row = T.iloc[a_dist_df.loc[min_index, "idx"]]
        min_distance = a_dist_df.loc[min_index, "Dist"]
        filled_row = fill_placeholders(b_row, T_row, b_row.index)
        filled_values.append(filled_row)
        min_indices.append(min_index)
        min_distances.append(min_distance)

        print(f'For b.csv row {b_index}, minimum Hamming distance {min_distance} is at a.csv row {min_index}')

    # 最小のハミング距離を持つ行番号をcsvファイルとして出力
    output_df = pd.DataFrame({
        'a_index': min_indices,
        'filled_values': filled_values
    })
    output_df.to_csv('E.csv', index=False, header=None)

    # plt.hist(min_distances)
    # plt.savefig('E.png')


if __name__ == "__main__":
    # コマンドライン引数を読みこむ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', type=str, help='ID for the prefixes (e.g., 32 for B32a, B32b, C32)')
    parser.add_argument('Ba_prefix', nargs='?', help='e.g., B32a', default=None)
    parser.add_argument('Bb_prefix', nargs='?', help='e.g., B32b', default=None)
    parser.add_argument('C_prefix', nargs='?', help='e.g., C32', default=None)
    parser.add_argument('--no_print', action='store_true')

    args = parser.parse_args()

    # --id オプションが指定された場合にプレフィックスを設定
    if args.id:
        a_prefix = f'B{args.id}a'
        b_prefix = f'B{args.id}b'
        c_prefix = f'C{args.id}'
    else:
        if args.Ba_prefix is None or args.Bb_prefix is None or args.C_prefix is None:
            print("Error: When --id is not specified, Ba_prefix, Bb_prefix, and C_prefix must be provided.")
            sys.exit(1)
        a_prefix = args.Ba_prefix
        b_prefix = args.Bb_prefix
        c_prefix = args.C_prefix

    # --no_print オプションが指定された場合に標準出力を無効化
    if args.no_print:
        with redirect_stdout(open(os.devnull, 'w')):
            main(a_prefix, b_prefix, c_prefix)
    else:
        main(a_prefix, b_prefix, c_prefix)