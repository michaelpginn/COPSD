"""Convert the COPSD-AfriMGSM HuggingFace dataset into the per-language
translated JSON files the training scripts expect.

For each language present in the dataset, writes:
    ./african_translated/translated_<lang>.json

Each output row has the fields the multilingual collator consumes:
    problem            -> English source problem (becomes problem_en at runtime)
    solution           -> English reference solution (teacher context)
    problem_<lang>     -> target-language problem (student sees this)
    Answer, source     -> carried over for reference
"""

import json
import os
from collections import defaultdict

from datasets import load_dataset

DATASET = "yihongLiu/COPSD-AfriMGSM-TrainDataset"
OUT_DIR = "./african_translated"


def main():
    ds = load_dataset(DATASET)

    # Bucket rows by language across all splits.
    by_lang = defaultdict(list)
    for split in ds:
        for ex in ds[split]:
            lang = ex["language"]
            by_lang[lang].append(
                {
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    f"problem_{lang}": ex["problem_translated"],
                    "Answer": ex.get("Answer"),
                    "source": ex.get("source"),
                }
            )

    os.makedirs(OUT_DIR, exist_ok=True)

    for lang in sorted(by_lang):
        rows = by_lang[lang]
        out_path = os.path.join(OUT_DIR, f"translated_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] wrote {len(rows)} rows -> {out_path}")

    print(f"\nDone. {len(by_lang)} languages written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
