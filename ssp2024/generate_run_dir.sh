#!/bin/sh

# 最も大きいランディレクトリ番号を探す
last_run=$(ls -d run[0-9][0-9][0-9] 2>/dev/null | sort | tail -n 1)

if [ -z "$last_run" ]; then
  # もしランディレクトリが存在しない場合、run001を作成する
  new_run="run001"
else
  # 既存のランディレクトリ番号に+1
  num=$(echo $last_run | sed 's/run//')
  new_num=$(printf "%03d" $((num + 1)))
  new_run="run$new_num"
fi

# 新しいランディレクトリを作成
mkdir $new_run

# hweq2dから必要なファイルをコピー
cp -r hweq2d/data hweq2d/diag hweq2d/param.namelist $new_run/

# hweq2d.exeは相対パスでシンボリックリンクを作成
ln -s ../hweq2d/hweq2d.exe "$new_run/hweq2d.exe"

echo "New run directory created: $new_run"
