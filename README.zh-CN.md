# 金融 NLP — 深度学习实战系列

[English](README.md) | **简体中文**

一套注释详尽的实战 notebook，围绕**金融情感分析**这一主线，依次实现文本深度学习的
主要架构（**CNN、RNN/LSTM/GRU、Transformer/BERT**），并附带一个**股价预测**的
时序小例子。

> **关于本仓库**
> 本仓库为**自学 / 作品集**项目。模型设计遵循下方引用的经典论文，目标是提供清晰、
> 可读、端到端的实现与架构对比，而非原创研究。

---

## 内容一览

| # | Notebook | 主题 | 要点 |
|---|----------|------|------|
| 01 | `01_textcnn_sentiment.ipynb` | **TextCNN** 金融情感分类 | Kim (2014) 多核 CNN + 冻结的预训练 word2vec；在 FinancialPhraseBank 上做三分类 |
| 02 | `02_textrnn_sentiment.ipynb` | **RNN 家族**对比 | RNN / LSTM / GRU / 双向 GRU / 三层双向 GRU；打包变长序列；准确率与耗时对比图 |
| 03 | `03_stock_price_gru_tcn.ipynb` | **股价预测** | 多股日频特征的滚动窗口 `Dataset` → 预测次日收盘；**GRU vs 时序卷积网络 (TCN)**；R² 评估 |
| 04 | `04_bert_finetune_bertopic.ipynb` | **BERT 微调 + 主题建模** | 微调 `bert-base-uncased`（全量 vs 仅分类头）做情感分类；用 MiniLM 嵌入跑 **BERTopic** 聚类 |
| 05 | `05_chinese_bert.ipynb` | **中文 BERT** | 用 MarianMT 把语料英译中，再微调 `chinese-roberta-wwm-ext` 做中文情感分类 |

> 主题数据集：**FinancialPhraseBank**（Malo et al., 2014）——约 4840 条金融新闻句子，
> 标注为 positive / neutral / negative。

---

## 目录结构

```
financial-nlp/
├── 01_textcnn_sentiment.ipynb
├── 02_textrnn_sentiment.ipynb
├── 03_stock_price_gru_tcn.ipynb
├── 04_bert_finetune_bertopic.ipynb
├── 05_chinese_bert.ipynb
├── environment.yml               # conda/mamba 环境
├── requirements.txt              # pip 备选
├── scripts/
│   └── download_models.py        # 下载预训练模型到 ./model/
├── data/                         # 数据集放这里（见 data/README.md）——不纳入版本管理
│   └── README.md
└── model/                        # 模型权重放这里（见 model/README.md）——不纳入版本管理
    └── README.md
```

数据集与模型权重**不纳入版本管理**（体积大且有许可限制），详见
[`data/README.md`](data/README.md) 与 [`model/README.md`](model/README.md)。

---

## 环境配置

```bash
git clone <your-repo-url>
cd financial-nlp
```

**方案 A — conda / mamba**（推荐，使用 `environment.yml`）：

```bash
mamba env create -f environment.yml      # 或：conda env create -f environment.yml
mamba activate nlp
```

**方案 B — pip + venv**：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**获取模型**（约 1.6 GB，只需一次）：

```bash
python scripts/download_models.py            # 全部模型
python scripts/download_models.py --mirror   # 走 hf-mirror.com（中国大陆）
```

**获取数据**：参见 [`data/README.md`](data/README.md)。FinancialPhraseBank 可免费下载；
股票数据（notebook 03）需要 RESSET 数据库权限。

---

## 运行

```bash
jupyter lab        # 或：jupyter notebook
```

按编号顺序运行。`01`、`02`、`04`、`05` 只需 FinancialPhraseBank 与对应模型；`03` 另需
RESSET 股票数据。代码会在可用时自动使用 CUDA，否则回退到 CPU。

---

## 技术栈

PyTorch · Hugging Face Transformers · Datasets · Evaluate ·
sentence-transformers · BERTopic · gensim (word2vec) · scikit-learn ·
pandas · matplotlib

---

## 参考文献

- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
  *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT.* NAACL.
- Cui, Y., et al. (2021). *Pre-Training with Whole Word Masking for Chinese BERT.* (chinese-roberta-wwm-ext)
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic
  Convolutional and Recurrent Networks for Sequence Modeling.* (TCN)
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF.*
- Tiedemann, J., & Thottingal, S. (2020). *OPUS-MT.* (MarianMT en→zh)

---

## 许可

源代码以 **MIT 协议**发布（见 [`LICENSE`](LICENSE)）。数据集与预训练模型不在此协议
范围内，各自遵循原许可——其中 **FinancialPhraseBank 为 CC BY-NC-SA 3.0（非商业）**，
**股票数据为 RESSET 商业数据（不可转载）**。
