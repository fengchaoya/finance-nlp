# Models

This folder is **intentionally empty** in the repository. Pretrained weights are
large (the full set is ~1.6 GB) and each carries its own license, so they are
**not committed to git**. Download them locally before running the notebooks.

## Quick start

```bash
# From the repo root
python scripts/download_models.py            # downloads everything below
python scripts/download_models.py --mirror   # use hf-mirror.com (for users in mainland China)
```

## Expected layout

After downloading, this folder should contain:

| Path | Source | Used by |
|------|--------|---------|
| `bert-base-uncased-model/`, `bert-base-uncased-tokenizer/` | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) (Apache-2.0) | `04_bert_finetune_bertopic.ipynb` |
| `all-MiniLM-L6-v2/` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0) | `04_bert_finetune_bertopic.ipynb` (BERTopic) |
| `chinese-roberta-wwm-ext/` | [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext) (Apache-2.0) | `05_chinese_bert.ipynb` |
| `opus-mt-en-zh/` | [`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) (CC-BY-4.0) | `05_chinese_bert.ipynb` (en→zh translation) |
| `GoogleNews-vectors-negative300-SLIM.bin` | [word2vec-slim](https://github.com/eyaler/word2vec-slim) (trimmed Google News word2vec) | `01_textcnn_sentiment.ipynb`, `02_textrnn_sentiment.ipynb` |

Each model's license belongs to its original publisher — see the links above.
