<p align="center">
  <img src="pics/tcrdesign.png" width="85%" alt="header">
</p>

---

简体中文 | [English](./README.md)

该仓库提供了 TcrDesign 推理流程的实现。

我们还提供了用于 TcrDesign 的权重和数据集。如需详细信息，请参阅[https://zenodo.org/records/14545852](https://zenodo.org/records/14545852).

## 主要模型

TcrDesign 由两个组件组成：**TcrDesign-B**（结合模型）和 **TcrDesign-G**（生成模型）。

**TcrDesign-B** 精确预测表位与 TCR 之间的相互作用，结合了 VJ 基因和 MHC 等全面信息，实现了领先的性能。

**TcrDesign-G** 有效生成大量针对特定表位的 TCR。

## 安装

首先，请下载代码仓库 `git clone https://github.com/XSLiuLab/TcrDesign`

```
1. conda create -n tcrdesign python=3.8.16 && conda activate tcrdesign
2. conda install numpy=1.23.5 pandas=1.5.3 scikit-learn=1.2.2 tqdm=4.65.0 editdistance Levenshtein
3. conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
4. pip install matplotlib seaborn tensorboard transformers
```

请从[https://zenodo.org/records/14545852](https://zenodo.org/records/14545852)下载 `tcrdesign_weights.tar.gz`，解压缩并将其内容放置在`'weights'`文件夹中 ​。

## 用法

### TcrDesign-B

#### 示例 1: 单样本结合预测

对于缺失值，请使用 `'X'`​ 作为占位符。

```
python TcrDesign/tcrdesign_B.py -mode single -alphav TRAV12-2 -alphaj X -alpha_cdr3 CAVRGTGRRALTF -betav TRBV6-6 -betaj X -beta_cdr3 CASSFATEAFF -epitope GLYDGMEHL -mhc HLA-A02:01 -cuda False
```

#### 示例 2: 批量样本结合预测

准备一个制表符分隔的文件，格式遵循 `TcrDesign/example/Binding_batch_example.tsv`​。使用 `TcrDesign/data/mhc_pseudo/mhc_all.dat`​ 作为参考，将缺失值替换为 `X`​。

```
python TcrDesign/tcrdesign_B.py -mode batch -data_path TcrDesign/example/Binding_batch_example.tsv -cuda True
```

### TcrDesign-G

#### 示例 1: 生成特定表位的 bCDR3

```
python TcrDesign/tcrdesign_G.py -mode beta -epitope GILGFVFTL -num 100 -maxLen 20 -cuda True
```

#### 示例 2: 生成特定表位的 aCDR3

```
python TcrDesign/tcrdesign_G.py -mode alpha -epitope GILGFVFTL -bcdr3 CASSIRSTYEQYF -num 100 -maxLen 20 -cuda True
```

#### 示例 3: 生成特定 CDR3 的 VJ

```
# aCDR3 for VJ
python TcrDesign/tcrdesign_G.py -mode vj -acdr3 CAVNQGAQKLVF -cuda True
# bCDR3 for VJ
python TcrDesign/tcrdesign_G.py -mode vj -bcdr3 CASSIRSTYEQYF -cuda True
```

### TcrDesign

#### 示例: 生成特定表位的全长 TCR

```
python TcrDesign/tcrdesign.py -epitope GILGFVFTL -mhc HLA-A02:01 -gen_beta_num 30 -gen_alpha_num 100 -cuda True
```

## 引用 TcrDesign

如果您在研究中使用 TcrDesign，请引用我们的论文。

---

**上海科技大学肿瘤生物学研究组**

**由上海科技大学刘雪松教授领导的研究团队进行研究**
