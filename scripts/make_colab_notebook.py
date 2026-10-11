"""
Build a self-contained Colab notebook from the repository sources.

Every Python file the pipeline needs is embedded as a %%writefile cell, so
the notebook shows all the code and does not depend on cloning the repo.

Run from the repository root:
    python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "train_colab.ipynb"
PROJ = "/content/drive/MyDrive/semantic-comm-transceiver"

# (path, section title, one-line explanation)
FILES = [
    ("requirements.txt", "依赖列表", "需要安装的 Python 包。"),
    ("config.py", "配置", "所有超参数、路径、评估 SNR 点、验证集比例、BERT 融合调参网格。"),
    ("utils.py", "工具函数", "随机种子、token id 转文本、混合精度类型选择。"),
    ("dataset.py", "数据加载", "读取 pkl 数据；从训练集固定划出 5% 作为验证集。"),
    ("models/__init__.py", "模型包入口", ""),
    ("models/channel.py", "信道模型",
     "AWGN 和 Rayleigh（E|h|²=1）信道，同时返回接收端的有效 SNR。"),
    ("models/modules.py", "网络模块",
     "SNR 自适应注意力、编码/解码层、FiLM（让信道编解码器感知 SNR）、"
     "只给有效 token 分配能量的功率归一化。"),
    ("models/transceiver.py", "收发机模型",
     "BaselineDeepSC 与 ImprovedDeepSC；共享的贪心解码、批量束搜索、候选词提取。"
     "Improved 的每项改进都可以单独关闭，用于消融实验。"),
    ("models/losses.py", "损失函数", "交叉熵 + 逐 token 特征去噪损失。"),
    ("models/llm_refiner.py", "BERT 融合修正",
     "只在解码器不确定的位置，用 BERT 在解码器自己的 top-5 候选词里重新打分。"),
    ("train.py", "训练脚本",
     "训练一个模型：学习率预热 + 余弦退火、按验证集 −5/0/5 dB 的平均准确率选 best、"
     "混合精度、断点续训。"),
    ("evaluate.py", "评估脚本",
     "在 7 个 SNR 下计算 WordAcc / BLEU（含 95% 置信区间）/ BERTScore / 句子相似度；"
     "--tune_llm 在验证集上调 BERT 融合参数。"),
    ("finetune_bert.py", "BERT 微调", "用 Europarl 训练句微调 BERT，供融合修正使用。"),
    ("ablation.py", "结果汇总", "读取所有评估结果，生成 summary.md 和曲线图。"),
    ("run_all.py", "总调度",
     "依次训练 8 个模型 → 评估 → 微调 BERT → 调参 → 出图。可重复运行：已完成的步骤会跳过。"),
]

SHA = {
    "sentences.pkl": "886bfe2761e394b20cbafe04080a74a6e1bcf083e792ae7d067a14bedc1b35c0",
    "test_data.pkl": "d61ad3589e07914223411acab00f9ee753156cacdd93b1d7da549a88785847da",
    "train_data.pkl": "581faa9be369b97041914fac5a8d2bcc3a29a63c0790d1f2404c1122b5169bcc",
    "vocab.pkl": "2493197f50eb289eaf9e4b2c40968f46ca12efe166659c00e7761ce4e2f06ec8",
}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def build():
    cells = [md("""# 语义通信收发机 — 完整训练笔记本

这个笔记本包含训练所需的**全部代码**。每个 `%%writefile` 单元格会把一个代码文件写到 Colab 机器的 `/content/sc/` 下，然后由最后的单元格调用。

**使用方法：** 先在菜单“代码执行程序 → 更改运行时类型”里选择 GPU，然后点“代码执行程序 → 全部运行”，或者从上到下依次运行。

**Drive 里的项目文件夹** `MyDrive/semantic-comm-transceiver/`：

| 路径 | 内容 |
|---|---|
| `data/` | 数据压缩包分片（第 2 节还原为 4 个 pkl） |
| `checkpoints/` | 模型权重、评估结果、`run_all.log` 日志（运行时生成） |
| `results/` | 汇总表 `summary.md` 和曲线图（运行时生成） |

**断线后：** 重新“全部运行”即可。已训练完的模型会跳过，训练到一半的会从 `last.pt` 继续，已有的评估结果不会重算。

**实验内容：** 8 个模型 × 80 epoch。AWGN 信道：Baseline、Improved 和 3 个消融版本（去掉 FiLM / 去掉带掩码的功率归一化 / 去掉特征去噪损失）；Rayleigh 信道：Baseline、Improved 和 1 个消融版本（去掉接收端有效 SNR）。T4 上训练约 16 小时，A100/L4 会快很多。"""),
             md("## 1. 准备工作目录、挂载 Drive"),
             code(f"""from google.colab import drive
drive.mount('/content/drive')

import os
PROJ = '{PROJ}'
os.makedirs('/content/sc/models', exist_ok=True)
os.makedirs('/content/sc/data', exist_ok=True)
%cd /content/sc""")]

    cells.append(md("## 2. 代码文件\n\n下面每个单元格写出一个文件。改进相关的核心代码在 `models/` 下的几个文件里。"))
    for path, title, desc in FILES:
        src = (ROOT / path).read_text(encoding="utf-8").rstrip("\n")
        cells.append(md(f"### `{path}` — {title}\n\n{desc}" if desc else f"### `{path}` — {title}"))
        cells.append(code(f"%%writefile {path}\n{src}"))

    sha_lines = "\n".join(f"    '{k}': '{v}'," for k, v in SHA.items())
    cells += [
        md("## 3. 数据：合并 Drive 上的分片并校验"),
        code(f"""!cat "$PROJ"/data/sc_data.tar.gz.part_* | tar xz -C data
!ls -lh data

import hashlib
expected = {{
{sha_lines}
}}
ok = True
for name, sha in expected.items():
    h = hashlib.sha256(open(f'data/{{name}}', 'rb').read()).hexdigest()
    ok &= h == sha
    print(f'{{name:16s}}', 'OK' if h == sha else 'MISMATCH')
assert ok, '数据和本地文件不一致'"""),
        md("## 4. 安装依赖、检查 GPU"),
        code("""!pip install -q -r requirements.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), '没有 GPU：请在“更改运行时类型”里选择 GPU'
print('bf16 (A100/L4 等):', torch.cuda.get_device_capability()[0] >= 8)"""),
        md("## 5. 快速自检（约 5–10 分钟）\n\n每个模型只训练 20 个 batch、只评估 64 句，确认整条流程能跑通。输出写在 Colab 本地的 `smoke_checkpoints/`，不会影响 Drive 里的正式结果。"),
        code("""os.environ['TQDM_DISABLE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
!python run_all.py --smoke 2>&1 | grep -E "Running:|\\[Epoch|Results saved|Best lambda|All done|Error|Traceback" """),
        md("## 6. 正式训练和评估\n\n全部结果写到 Drive。完整日志在 `checkpoints/run_all.log`；这里只显示每个 epoch 的汇总和各步骤的开始/结束。"),
        code("""os.environ['SC_CKPT_DIR'] = f'{PROJ}/checkpoints'
os.environ['SC_RESULT_DIR'] = f'{PROJ}/results'
os.environ['TQDM_DISABLE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.makedirs(os.environ['SC_CKPT_DIR'], exist_ok=True)
!python run_all.py --amp 2>&1 | tee -a "$PROJ/checkpoints/run_all.log" | grep --line-buffered -E "Running:|Model:|\\[Epoch|Training complete|Results saved|Best lambda|already|skipping|All done|Error|Traceback" """),
        md("## 7. 查看结果"),
        code("""RESULTS = f'{PROJ}/results'
from IPython.display import Markdown, Image, display
import glob
display(Markdown(open(f'{RESULTS}/summary.md').read()))
for png in sorted(glob.glob(f'{RESULTS}/*BLEU-4*.png')) + [f'{RESULTS}/training_curves.png']:
    display(Image(png, width=700))"""),
    ]

    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(cells)} cells)")


if __name__ == "__main__":
    build()
