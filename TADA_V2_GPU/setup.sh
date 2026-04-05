#!/usr/bin/env bash
#
# TADA_T2 GPU 一键安装脚本
#
# 用法:
# chmod +x setup.sh && ./setup.sh
#
set -euo pipefail

INSTALL_DIR="${1:-.}"
cd "$INSTALL_DIR"

echo "═══════════════════════════════════════════════════════════"
echo " TADA_T2 GPU 加速版 — 一键安装"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: 克隆原始仓库 ────────────────────────────────────
if [ -d "TADA_T2" ]; then
 echo "⚠️ TADA_T2 目录已存在，跳过克隆"
else
 echo "📥 克隆 TADA_T2 原始仓库..."
 git clone --depth 1 https://github.com/ryanemenecker/TADA_T2.git
 echo " ✅ 克隆完成"
fi
echo ""

# ── Step 2: 安装 Python 依赖 ────────────────────────────────
echo "📦 安装依赖..."
pip install -q tensorflow numpy alphaPredict protfasta tqdm 2>/dev/null || \
pip install tensorflow numpy alphaPredict protfasta tqdm
echo " ✅ 依赖安装完成"
echo ""

# ── Step 3: 安装 TADA_T2 包 ─────────────────────────────────
echo "🔧 安装 TADA_T2 包..."
cd TADA_T2
pip install -e . 2>/dev/null || pip install -e .
cd ..
echo " ✅ TADA_T2 已安装"
echo ""

# ── Step 4: 覆盖修改后的文件 ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取已安装包的 backend 目录
BACKEND_DIR=$(python3 -c 'import TADA_T2; import os; print(os.path.join(os.path.dirname(TADA_T2.__file__), "backend"))')

for FILE in features.py model.py predictor.py; do
 if [ -f "$SCRIPT_DIR/src/TADA_T2/backend/$FILE" ]; then
  echo "🔄 替换 $FILE（GPU 加速版）..."
  cp "$SCRIPT_DIR/src/TADA_T2/backend/$FILE" "$BACKEND_DIR/$FILE"
  echo " ✅ $FILE 已替换"
 fi
done
echo ""

# ── 完成 ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " ✅ 安装完成！"
echo ""
echo " 用法："
echo " cp $SCRIPT_DIR/src/predict_tad.py /your/work/dir/"
echo " cd /your/work/dir"
echo " python3 predict_tad.py sequences.fasta -o results.tsv"
echo ""
echo " 查找表会在首次运行时自动生成（如缺失）。"
echo " 也可以手动指定："
echo " python3 predict_tad.py seq.fa --alpha_lookup ./a.npy --cider_lookup ./c.npy"
echo "═══════════════════════════════════════════════════════════"
