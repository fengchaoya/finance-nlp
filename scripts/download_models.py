#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download the pretrained models the notebooks expect into ./model/.

Usage:
    python scripts/download_models.py            # download everything
    python scripts/download_models.py --mirror   # via hf-mirror.com (mainland China)
    python scripts/download_models.py --only bert word2vec

Models are large (~1.6 GB total); the download runs once and is then cached
locally. Re-running skips files that already exist.
"""
import argparse
import gzip
import os
import shutil
import sys
import urllib.request

# repo_root/model  (this file lives in repo_root/scripts)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")

WORD2VEC_URL = (
    "https://github.com/eyaler/word2vec-slim/raw/master/"
    "GoogleNews-vectors-negative300-SLIM.bin.gz"
)


def _exists(path):
    return os.path.exists(path) and (
        not os.path.isdir(path) or len(os.listdir(path)) > 0
    )


def download_bert():
    """bert-base-uncased: saved as separate model/ and tokenizer/ dirs."""
    from transformers import BertForSequenceClassification, BertTokenizer

    tok_dir = os.path.join(MODEL_DIR, "bert-base-uncased-tokenizer")
    mdl_dir = os.path.join(MODEL_DIR, "bert-base-uncased-model")
    if _exists(tok_dir) and _exists(mdl_dir):
        print("  [skip] bert-base-uncased already present")
        return
    print("  downloading bert-base-uncased ...")
    BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained(tok_dir)
    BertForSequenceClassification.from_pretrained("bert-base-uncased").save_pretrained(mdl_dir)


def download_minilm():
    """all-MiniLM-L6-v2 sentence embedding model (for BERTopic)."""
    from sentence_transformers import SentenceTransformer

    out = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")
    if _exists(out):
        print("  [skip] all-MiniLM-L6-v2 already present")
        return
    print("  downloading all-MiniLM-L6-v2 ...")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(out)


def download_chinese_roberta():
    """hfl/chinese-roberta-wwm-ext."""
    from transformers import AutoModel, AutoTokenizer

    out = os.path.join(MODEL_DIR, "chinese-roberta-wwm-ext")
    if _exists(out):
        print("  [skip] chinese-roberta-wwm-ext already present")
        return
    print("  downloading chinese-roberta-wwm-ext ...")
    AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext").save_pretrained(out)
    AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext").save_pretrained(out)


def download_opus():
    """Helsinki-NLP/opus-mt-en-zh translation model."""
    from transformers import MarianMTModel, MarianTokenizer

    out = os.path.join(MODEL_DIR, "opus-mt-en-zh")
    if _exists(out):
        print("  [skip] opus-mt-en-zh already present")
        return
    print("  downloading opus-mt-en-zh ...")
    MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh").save_pretrained(out)
    MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh").save_pretrained(out)


def download_word2vec():
    """Trimmed GoogleNews word2vec (.bin), gunzipped from GitHub."""
    out = os.path.join(MODEL_DIR, "GoogleNews-vectors-negative300-SLIM.bin")
    if _exists(out):
        print("  [skip] word2vec slim already present")
        return
    gz = out + ".gz"
    print("  downloading GoogleNews-vectors-negative300-SLIM (~50 MB gz) ...")
    urllib.request.urlretrieve(WORD2VEC_URL, gz)
    print("  decompressing ...")
    with gzip.open(gz, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz)


TASKS = {
    "bert": download_bert,
    "minilm": download_minilm,
    "chinese": download_chinese_roberta,
    "opus": download_opus,
    "word2vec": download_word2vec,
}


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models into ./model/")
    parser.add_argument("--mirror", action="store_true",
                        help="use https://hf-mirror.com (helpful in mainland China)")
    parser.add_argument("--only", nargs="+", choices=list(TASKS),
                        help="download only the named models (default: all)")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using Hugging Face mirror: https://hf-mirror.com")

    os.makedirs(MODEL_DIR, exist_ok=True)
    selected = args.only or list(TASKS)
    print(f"Target directory: {MODEL_DIR}")
    print(f"Downloading: {', '.join(selected)}\n")

    for name in selected:
        print(f"[{name}]")
        try:
            TASKS[name]()
        except Exception as exc:  # keep going on partial failure
            print(f"  [error] {name}: {exc}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
