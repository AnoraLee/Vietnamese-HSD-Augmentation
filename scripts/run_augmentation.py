from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import setup_logging
from src.utils.config import load_config, resolve_path
from src.utils.constants import LABELS, label2id


def run_back_translation(cfg: dict) -> None:
    """Dùng Helsinki-NLP qua transformers pipeline (offline-friendly)."""
    from transformers import pipeline

    bt_cfg = cfg["augmentation"]["back_translation"]
    src = bt_cfg.get("source_language", "vi")
    mid = bt_cfg.get("intermediate_language", "en")

    forward = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src}-{mid}")
    backward = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{mid}-{src}")

    train_df = pd.read_csv(resolve_path(cfg["data"]["train_file"]))
    if train_df["label"].dtype == object:
        train_df["label"] = train_df["label"].map(label2id)

    rows: list[dict] = []
    for label_name, n in bt_cfg["samples_per_label"].items():
        label_id = label2id[label_name]
        pool = train_df[train_df["label"] == label_id]
        if pool.empty:
            print(f"[bt] không có mẫu {label_name}, bỏ qua.")
            continue
        sample = pool.sample(n=min(n, len(pool)), random_state=42)
        for text in sample["text"]:
            try:
                en = forward(text, max_length=128)[0]["translation_text"]
                vi = backward(en, max_length=128)[0]["translation_text"]
                rows.append({"text": vi, "label": label_id})
            except Exception as exc:
                print(f"[bt] lỗi 1 mẫu: {exc}")

    out_path = resolve_path(bt_cfg["output_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[bt] saved {len(rows)} mẫu → {out_path}")


def _eda_swap(words: list[str], alpha: float) -> list[str]:
    n = max(1, int(alpha * len(words)))
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def _eda_delete(words: list[str], alpha: float) -> list[str]:
    if len(words) <= 1:
        return words
    n = max(1, int(alpha * len(words)))
    for _ in range(n):
        if len(words) <= 1:
            break
        words.pop(random.randrange(len(words)))
    return words


def _eda_insert(words: list[str], alpha: float) -> list[str]:
    if not words:
        return words
    n = max(1, int(alpha * len(words)))
    for _ in range(n):
        words.insert(random.randrange(len(words) + 1), random.choice(words))
    return words


EDA_OPS = {"random_swap": _eda_swap, "random_deletion": _eda_delete, "random_insertion": _eda_insert}


def run_eda(cfg: dict) -> None:
    eda_cfg = cfg["augmentation"]["eda"]
    seed = cfg["augmentation"].get("seed", 42)
    random.seed(seed)

    train_df = pd.read_csv(resolve_path(cfg["data"]["train_file"]))
    if train_df["label"].dtype == object:
        train_df["label"] = train_df["label"].map(label2id)

    ops = eda_cfg["operations"]
    alpha = eda_cfg["alpha"]
    num_aug = eda_cfg["num_aug"]

    rows: list[dict] = []
    for label_name, n in eda_cfg["samples_per_label"].items():
        label_id = label2id[label_name]
        pool = train_df[train_df["label"] == label_id]
        if pool.empty:
            continue
        sample = pool.sample(n=min(n, len(pool)), random_state=seed)
        for text in sample["text"]:
            for _ in range(num_aug):
                op = random.choice(ops)
                words = str(text).split()
                new_words = EDA_OPS[op](words.copy(), alpha)
                rows.append({"text": " ".join(new_words), "label": label_id})

    out_path = resolve_path(eda_cfg["output_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[eda] saved {len(rows)} mẫu → {out_path}")


def run_llm(cfg: dict) -> None:
    """Gọi LLM để rewrite câu giữ nguyên nhãn.

    Cần provider API — mặc định dùng OpenAI-compatible endpoint.
    Set env: LLM_API_KEY, LLM_BASE_URL (tùy chọn).
    """
    import json
    import os

    llm_cfg = cfg["augmentation"]["llm"]
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("[llm] không có LLM_API_KEY, bỏ qua.")
        return

    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    train_df = pd.read_csv(resolve_path(cfg["data"]["train_file"]))
    if train_df["label"].dtype == object:
        train_df["label"] = train_df["label"].map(label2id)

    max_samples = llm_cfg["max_samples"]
    prompt_template = llm_cfg["prompt_template"]
    id2label = {i: l for l, i in label2id.items()}

    log_path = resolve_path(llm_cfg["prompt_log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")

    pool = train_df[train_df["label"].isin([label2id["OFFENSIVE"], label2id["HATE"]])]
    sample = pool.sample(n=min(max_samples, len(pool)), random_state=42)

    rows: list[dict] = []
    for i, row in enumerate(sample.itertuples(), 1):
        label_name = id2label[row.label]
        prompt = prompt_template.format(label_name=label_name, text=row.text)
        log_f.write(json.dumps({"i": i, "label": label_name, "prompt": prompt}, ensure_ascii=False) + "\n")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=256,
            )
            new_text = resp.choices[0].message.content.strip()
            rows.append({"text": new_text, "label": row.label})
        except Exception as exc:
            print(f"[llm] lỗi mẫu {i}: {exc}")

    log_f.close()

    for provider, path in llm_cfg["output_files"].items():
        out_path = resolve_path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[llm:{provider}] saved {len(rows)} mẫu → {out_path}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None,
                        choices=["bt", "eda", "llm"],
                        help="Chỉ chạy các method chỉ định. Mặc định: chạy hết enabled.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    aug = cfg["augmentation"]

    run_all = args.methods is None

    if (run_all or "bt" in args.methods) and aug["back_translation"].get("enabled"):
        run_back_translation(cfg)
    if (run_all or "eda" in args.methods) and aug["eda"].get("enabled"):
        run_eda(cfg)
    if (run_all or "llm" in args.methods) and aug["llm"].get("enabled"):
        run_llm(cfg)

    print("[done] augmentation hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())