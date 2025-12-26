#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def load_urls(csv_path: str, url_col: str = "url") -> List[str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Failed to read {csv_path}: {e}", file=sys.stderr)
        return []
    if url_col not in df.columns:
        print(f"[!] Column '{url_col}' not found in {csv_path}. Available: {list(df.columns)}", file=sys.stderr)
        return []
    urls = df[url_col].astype(str).fillna("").tolist()
    # 基本清理：去除首尾空白
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip() != ""]
    return urls


def compute_lengths(urls: List[str], tokenizer, batch_size: int = 512) -> np.ndarray:
    lengths = []
    for i in tqdm(range(0, len(urls), batch_size), desc="Tokenizing", unit="batch"):
        batch = urls[i:i + batch_size]
        # 不截斷，量測自然長度
        enc = tokenizer(
            batch,
            padding=False,
            truncation=False,
            add_special_tokens=True,  # 與訓練時一致（通常會加 CLS/SEP）
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # enc["input_ids"] 是 list[list[int]]
        batch_lengths = [len(ids) for ids in enc["input_ids"]]
        lengths.extend(batch_lengths)
    return np.array(lengths, dtype=np.int32)


def summarize(lengths: np.ndarray) -> Dict[str, float]:
    if lengths.size == 0:
        return {}
    stats = {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        ">64_ratio": float((lengths > 64).mean()),
        ">=64_ratio": float((lengths >= 64).mean()),
        ">128_ratio": float((lengths > 128).mean()),
        ">256_ratio": float((lengths > 256).mean()),
        ">512_ratio": float((lengths > 512).mean()),
    }
    return stats


def print_report(name: str, stats: Dict[str, float], model_name: str, max_pos: int):
    print("\n==============================")
    print(f"File: {name}")
    print(f"Model: {model_name}")
    print(f"model.max_position_embeddings: {max_pos}")
    if not stats:
        print("No data.")
        return
    print("Count:", stats["count"])
    print(f"Min/Max: {stats['min']} / {stats['max']}")
    print("Percentiles:")
    print(f"  P50: {stats['p50']:.1f}  P75: {stats['p75']:.1f}  P90: {stats['p90']:.1f}  P95: {stats['p95']:.1f}  P99: {stats['p99']:.1f}")
    print("Ratios (proportion of samples longer than threshold):")
    print(f"  > 64:  {stats['>64_ratio']:.3f}    (>=64: {stats['>=64_ratio']:.3f})")
    print(f"  > 128: {stats['>128_ratio']:.3f}")
    print(f"  > 256: {stats['>256_ratio']:.3f}")
    print(f"  > 512: {stats['>512_ratio']:.3f}")
    if max_pos is not None:
        over_model = float((stats['count'] > 0) and 0)  # placeholder


def main():
    parser = argparse.ArgumentParser(description="Check token length distribution of URLs in CSV files.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="CSV files to check (must contain a 'url' column).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("HF_MODEL", "CrabInHoney/urlbert-tiny-v4-phishing-classifier"),
        help="Hugging Face tokenizer/model name (default from HF_MODEL env or urlbert-tiny-v4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for tokenization.",
    )
    parser.add_argument(
        "--url-col",
        type=str,
        default="url",
        help="Column name for URLs (default: url).",
    )
    args = parser.parse_args()

    # 載入 tokenizer
    print(f"[Info] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_pos = getattr(getattr(tokenizer, "model_max_length", None), "__int__", lambda: None)()
    # 更準確的是 model.config.max_position_embeddings，但這裡我們只拿 tokenizer 可得的
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        max_pos = getattr(cfg, "max_position_embeddings", max_pos)
    except Exception:
        pass

    for f in args.files:
        urls = load_urls(f, url_col=args.url_col)
        if not urls:
            print(f"[!] Skip {f}: no URLs.")
            continue
        lengths = compute_lengths(urls, tokenizer, batch_size=args.batch_size)
        stats = summarize(lengths)
        print_report(os.path.basename(f), stats, args.model, max_pos)

    print("\n[Done] If many samples exceed model.max_position_embeddings, consider:")
    print("- Cleaning/normalizing URLs to keep key parts early (domain, first path segments, short query keys).")
    print("- Using a model with larger max_position_embeddings and increasing max_len accordingly.")


if __name__ == "__main__":
    main()