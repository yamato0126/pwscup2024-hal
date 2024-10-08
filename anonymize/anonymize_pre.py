"""

予備戦の匿名化プログラム. 使い方はサンプル匿名化プログラムと同じ.

"""

import os
import random
import argparse
import warnings
import re

import pandas as pd
import numpy as np

all_columns = [
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

swap_columns = [
    [['Gender', 'Age'], ['Occupation', '260'], ['653', '1525'], ['2105', '2193'], ['2253', '2628'], ['2872', '3438'], ['3439', '3440'], ['3877', '3889'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation', '2'], ['56', '260'], ['653', '673'], ['1009', '1073'], ['1525', '1750'], ['1881', '1967'], ['2043', '2093'], ['2105', '2143'], ['2193', '2399'], ['2628', '2968'], ['3479', '3489'], ['3877', '3889'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation', '673'], ['1881', '1920'], ['2087', '2138'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation'], ['2', '56'], ['673', '810'], ['885', '1009'], ['1073', '1097'], ['1525', '1654'], ['1702', '1750'], ['1881', '1920'], ['1967', '2017'], ['2043', '2087'], ['2093', '2138'], ['2399', '3438'], ['3439', '3440'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation', '673'], ['810', '1073'], ['1126', '1702'], ['2100', '2174'], ['2253', '2797'], ['3393', '3466'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation', '247'], ['885', '1097'], ['1654', '2086'], ['2138', '2872'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation', '247'], ['2100', '2143'], ['2872', '3479'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation'], ['260', '1097'], ['1750', '2021'], ['2093', '2105'], ['2628', '2968'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation'], ['247', '1920'], ['2017', '2087'], ['ZIP-code']],
    [['Gender', 'Age'], ['Occupation'], ['260', '1097'], ['2628', '2174'], ['2797', '1073'], ['2100', '2968'], ['2105', '2193'], ['ZIP-code']]
]

# 匿名化処理を書いた関数
# ここを変更すると匿名化の方法を変えられる
# pandasのデータフレームを入力することを想定
def anonymize(df, subset_id):
    anonymized_df = df.copy()

    # 加工処理の適用
    print("1. Name列の削除")
    if 'Name' in anonymized_df.columns:
        # 'Name'列は使わないので、(もしあれば)削除する
        anonymized_df.drop('Name', axis=1, inplace=True)
    
    print("2. group_columns が一致しているユーザの中で、swap_columns をシャッフル")
    num_split = len(swap_columns[subset_id])
    len_split_df = len(anonymized_df) // (num_split + 3) # ZIP-code のみ4倍のデータ
    anonymized_df_list = []
    for i in range(num_split):
        group_columns = swap_columns[subset_id][i]
        target_columns = list(set(all_columns[subset_id]) - set(group_columns))
        if i < num_split - 1:
            tmp_df = anonymized_df[i*len_split_df:(i+1)*len_split_df]
        else:
            tmp_df = anonymized_df[i*len_split_df:]
        print(f"group_columns: {group_columns}")
        print(f"len of tmp_df: {len(tmp_df)}")
        anonymized_df_list.append(group_shuffle(tmp_df, group_columns, target_columns))
    anonymized_df = pd.concat(anonymized_df_list)

    print("3. ランダムに選んだ1列で、値を1000回入れ替え")
    anonymized_df = random_shuffle(anonymized_df, rep=1000)

    print("4. ランダムに行をシャッフル")
    anonymized_df = anonymized_df.sample(frac=1)

    return anonymized_df

def random_shuffle(df, rep=1000):
    # ランダムに選んだ列の中でセルの入れ替え　x 1000回
    anonymized_df = df.copy()
    for _ in range(rep):
        col_name = random.choice(df.columns.tolist())
        users = np.random.choice(np.arange(len(df)), 2, replace=False)
        tmp = anonymized_df.loc[users[0], col_name]
        anonymized_df.loc[users[0], col_name] = anonymized_df.loc[users[1], col_name]
        anonymized_df.loc[users[1], col_name] = tmp

    return anonymized_df

def group_shuffle(df, groups, targets):
    # group列で指定した列の値が同じ行内で、targets列の値をシャッフルする
    anonymized_df = df.copy()

    print(f"mean of groupby: {int(df.groupby(groups).size().mean())}")
    print(f"min of groupby: {int(df.groupby(groups).size().min())}")

    # Iterate over each group
    for name, group_data in anonymized_df.groupby(groups):
        if len(group_data) == 1:
            # If the group has only one row, we cannot shuffle the target column
            for target in targets:
                anonymized_df.loc[group_data.index, target] = random.choice(df[target].values)
        else:
            # Shuffle the target column within the current group
            for target in targets:
                shuffled_values = np.random.permutation(group_data[target].values)
                anonymized_df.loc[group_data.index, target] = shuffled_values

    return anonymized_df

def generate_output_filename(input_file_path, output_dir=None):
    # 正規表現パターン：Bから始まり、2桁の数字が続き、_と1桁の数字、またはBから始まり2桁の数字が続く
    pattern1 = re.compile(r'B(\d{2}_\d)\.csv$')
    pattern2 = re.compile(r'B(\d{2})\.csv$')
    
    match1 = pattern1.search(input_file_path)
    match2 = pattern2.search(input_file_path)
    
    if match1:
        output_filename = re.sub(r'^B', 'C', match1.group(0))
    elif match2:
        output_filename = re.sub(r'^B', 'C', match2.group(0))
    else:
        warnings.warn("想定していないファイル名です")
        output_filename = "C.csv"
    
    if output_dir:
        return os.path.join(output_dir, output_filename)
    else:
        input_dir = os.path.dirname(input_file_path)
        return os.path.join(input_dir, output_filename)

# メインの実行部分
if __name__ == "__main__":
    # コマンドライン引数を読みこむ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('org_csv', help='匿名化したいcsvファイル(e.g., B32_3.csv)')
    parser.add_argument('--output', '-o', help='出力ディレクトリ（指定しない場合は入力ファイルと同じディレクトリ）')
    args = parser.parse_args()

    # CSVファイルパスの読み込み
    input_file_path = args.org_csv

    # サブセットIDの取得 (e.g., B32_3.csv -> 3)
    subset_id = int(re.search(r'B\d+_(\d+)\.csv', input_file_path).group(1))

    # 出力ディレクトリの取得
    output_dir = args.output

    # 匿名化ファイル名の決定
    output_file_path = generate_output_filename(input_file_path, output_dir)

    # CSVファイルをpandasのデータフレームとして読み込む
    # 不正なファイルパスを指定するとここで強制終了
    Bi = pd.read_csv(input_file_path)

    # 匿名化処理
    Ci = anonymize(Bi, subset_id)

    # 結果の出力
    Ci.to_csv(output_file_path, index=False)
    print(f"匿名化ファイルを{output_file_path}に保存しました")

