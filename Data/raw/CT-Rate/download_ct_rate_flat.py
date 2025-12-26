#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download CT-RATE volumes listed in an Excel file (column: VolumeName)
and save the .nii.gz files directly into the same folder as this script
(flat output: no data_volumes/ and no nested subfolders).

It will:
- Read VolumeName list from Excel
- Try primary split then fallback splits
- Download via huggingface_hub to cache, then copy the file to output_dir
"""

import os
import argparse
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "ibrahimhamamci/CT-RATE"
REPO_TYPE = "dataset"


def build_subfolder(split: str, volume_name: str) -> str:
    """
    CT-RATE v2 structure (as observed):
    dataset/{split}/train_1000/train_1000_a/train_1000_a_1.nii.gz

    volume_name format: train_<id>_<series>_<idx>.nii.gz
    """
    base = volume_name.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected VolumeName format: {volume_name}")

    folder1 = parts[0]              # 'train'
    folder2 = parts[1]              # '1000'
    folder3 = parts[2]              # 'a'
    folder = f"{folder1}_{folder2}" # 'train_1000'
    subfolder = f"{folder}_{folder3}"  # 'train_1000_a'
    return f"dataset/{split}/{folder}/{subfolder}"


def try_download_flat(volume_name: str, splits_to_try, token: str | None,
                      cache_dir: Path, out_dir: Path, overwrite: bool):
    tried_paths = []
    last_err = None

    for split in splits_to_try:
        try:
            subfolder = build_subfolder(split, volume_name)
            rel = f"{subfolder}/{volume_name}"
            tried_paths.append(rel)

            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
                subfolder=subfolder,
                filename=volume_name,
                cache_dir=str(cache_dir),
                resume_download=True,
            )

            dest = out_dir / volume_name
            if dest.exists() and not overwrite:
                return True, tried_paths, "Skipped (already exists)"

            shutil.copy2(cached_path, dest)
            return True, tried_paths, None

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    return False, tried_paths, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Excel file containing a 'VolumeName' column")
    ap.add_argument("--sheet", default=None, help="Sheet name (optional)")
    ap.add_argument("--split", default="train_fixed", help="Primary split to try first (e.g., train_fixed)")
    ap.add_argument("--fallback", default="train,valid_fixed,valid", help="Comma-separated fallback splits")
    ap.add_argument("--start_at", type=int, default=0, help="Start index in the list")
    ap.add_argument("--batch_size", type=int, default=50, help="Batch size for progress grouping")
    ap.add_argument("--out_dir", default=None, help="Output folder. Default: folder containing this script")
    ap.add_argument("--cache_dir", default=".hf_cache", help="Cache folder (default: .hf_cache)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = (script_dir / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    # Read Excel
    # Read Excel (handle sheet_name=None -> dict of DataFrames)
    df_obj = pd.read_excel(args.excel, sheet_name=args.sheet)  # requires openpyxl

    if isinstance(df_obj, dict):
        # sheet=None => dict; pick the first sheet by default
        if len(df_obj) == 0:
            raise ValueError("Excel has no sheets.")
        first_sheet_name = next(iter(df_obj.keys()))
        df = df_obj[first_sheet_name]
        print(f"[Info] sheet_name not specified -> using first sheet: {first_sheet_name}")
    else:
        df = df_obj

    if "VolumeName" not in df.columns:
        raise ValueError(f"Excel must contain a column named 'VolumeName'. Found: {list(df.columns)}")

    volume_names = [str(x).strip() for x in df["VolumeName"].dropna().tolist()]

    total = len(volume_names)

    primary = args.split.strip()
    fallbacks = [s.strip() for s in args.fallback.split(",") if s.strip()]
    splits_to_try = [primary] + [s for s in fallbacks if s != primary]

    print(f"[Info] Total files: {total} | start_at={args.start_at} | batch_size={args.batch_size}")
    print(f"[Info] Primary split: {primary} | Fallback splits: {splits_to_try}")
    print(f"[Info] Output (flat): {out_dir}")
    print(f"[Info] Cache: {cache_dir}")

    success = 0
    failed = 0

    for i in tqdm(range(args.start_at, total, args.batch_size), desc="Batches"):
        batch = volume_names[i:i + args.batch_size]
        for name in batch:
            ok, tried, err = try_download_flat(
                volume_name=name,
                splits_to_try=splits_to_try,
                token=token,
                cache_dir=cache_dir,
                out_dir=out_dir,
                overwrite=args.overwrite,
            )
            if ok:
                success += 1
            else:
                failed += 1
                print(f"[Error] {name}: không tìm thấy / không tải được ở các đường dẫn sau:")
                for p in tried:
                    print(f"        - {p}")
                print(f"        Last error: {err}")

    print(f"[Done] Success={success} | Failed={failed}")
    if failed:
        print("Gợi ý: Nếu vẫn gặp 403, hãy kiểm tra token có bật quyền 'public gated repos'.")
        print("Nếu fail vài file, có thể VolumeName không tồn tại trong split bạn chọn (train_fixed vs train...).")


if __name__ == "__main__":
    main()
