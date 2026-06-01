# Financial NLP — Deep Learning Practice Series

**English** | [简体中文](README.zh-CN.md)

A hands-on, heavily-commented series of notebooks that work through the main
deep-learning architectures for text — **CNN, RNN/LSTM/GRU, Transformer (BERT)**
— on a consistent **financial sentiment analysis** theme, plus a short
time-series detour into **stock-price forecasting**.

> **About this repo**
> This is a **self-study / portfolio** project. The model designs follow
> well-known papers (cited below); the goal is clean, readable, end-to-end
> implementations and architecture comparisons — not novel research.

---

## What's inside

| # | Notebook | Topic | Highlights |
|---|----------|-------|------------|
| 01 | `01_textcnn_sentiment.ipynb` | **TextCNN** for financial sentiment | Kim (2014) multi-kernel CNN over frozen pretrained word2vec; 3-class classification on FinancialPhraseBank |
| 02 | `02_textrnn_sentiment.ipynb` | **RNN family** comparison | RNN vs LSTM vs GRU vs Bi-GRU vs 3-layer Bi-GRU; packed variable-length sequences; accuracy & runtime comparison plots |
| 03 | `03_stock_price_gru_tcn.ipynb` | **Stock-price forecasting** | Rolling-window `Dataset` over multi-stock daily features → next-day close; **GRU vs Temporal Convolutional Network (TCN)**; R² evaluation |
| 04 | `04_bert_finetune_bertopic.ipynb` | **BERT fine-tuning + topic modeling** | Fine-tune `bert-base-uncased` (full vs classifier-head-only) for sentiment; **BERTopic** clustering with MiniLM embeddings |
| 05 | `05_chinese_bert.ipynb` | **Chinese BERT** | MarianMT (en→zh) translation of the corpus, then fine-tune `chinese-roberta-wwm-ext` for Chinese sentiment |

> Theme dataset: **FinancialPhraseBank** (Malo et al., 2014) — ~4 840 financial
> news sentences labelled positive / neutral / negative.

---

## Repository layout

```
financial-nlp/
├── 01_textcnn_sentiment.ipynb
├── 02_textrnn_sentiment.ipynb
├── 03_stock_price_gru_tcn.ipynb
├── 04_bert_finetune_bertopic.ipynb
├── 05_chinese_bert.ipynb
├── environment.yml               # conda/mamba environment
├── requirements.txt              # pip alternative
├── scripts/
│   └── download_models.py        # fetch pretrained models into ./model/
├── data/                         # datasets go here (see data/README.md) — not committed
│   └── README.md
└── model/                        # model weights go here (see model/README.md) — not committed
    └── README.md
```

Datasets and model weights are **not committed** (large + licensed); see
[`data/README.md`](data/README.md) and [`model/README.md`](model/README.md).

---

## Setup

```bash
git clone <your-repo-url>
cd financial-nlp
```

**Option A — conda / mamba** (recommended; uses `environment.yml`):

```bash
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate nlp
```

**Option B — pip + venv**:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Get the models** (~1.6 GB, one-time):

```bash
python scripts/download_models.py            # all models
python scripts/download_models.py --mirror   # via hf-mirror.com (mainland China)
```

**Get the data**: follow [`data/README.md`](data/README.md).
FinancialPhraseBank is a free download; the stock data (notebook 03) requires
RESSET database access.

---

## Running

```bash
jupyter lab        # or: jupyter notebook
```

Run the notebooks in order. Notebooks `01`–`02` and `04`–`05` only need
FinancialPhraseBank + the relevant models; `03` additionally needs the RESSET
stock file. The code auto-selects CUDA when available and falls back to CPU.

---

## Tech stack

PyTorch · Hugging Face Transformers · Datasets · Evaluate ·
sentence-transformers · BERTopic · gensim (word2vec) · scikit-learn ·
pandas · matplotlib

---

## References

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

## License

Source code is released under the **MIT License** (see [`LICENSE`](LICENSE)).
Datasets and pretrained models are **not** covered by it and remain under their
own licenses — notably **FinancialPhraseBank is CC BY-NC-SA 3.0 (non-commercial)**
and the **stock data is proprietary (RESSET)**.
