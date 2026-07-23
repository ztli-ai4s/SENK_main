#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a fast filesystem-only index for qcMol .local files.

Why:
- 300k small .local files: glob + repeated HDF5 scans are expensive.
- For single-molecule-per-file datasets, we can build an index without opening HDF5.
- Optionally, we can *sample-validate* a subset of files by opening HDF5 and checking keys/schema.

Outputs:
- index JSONL: one record per file with path/size/mtime
- optional schema JSON: summary from sample validation

This does NOT parse molecule content; it is meant to speed up dataset initialization,
resume logic, and deterministic sharding for parallel preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class IndexRecord:
    path: str
    size: int
    mtime: float


def _iter_local_files(data_root: str) -> Iterable[IndexRecord]:
    # os.scandir is faster than glob for huge directories.
    with os.scandir(data_root) as it:
        for ent in it:
            if not ent.is_file():
                continue
            if not ent.name.endswith(".local"):
                continue
            try:
                st = ent.stat()
                yield IndexRecord(path=os.path.join(data_root, ent.name), size=int(st.st_size), mtime=float(st.st_mtime))
            except FileNotFoundError:
                continue


def _write_jsonl(records: List[IndexRecord], out_path: str) -> None:
    tmp = out_path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def _sample_validate(
    records: List[IndexRecord],
    sample_n: int,
    required_keys: Tuple[str, ...],
    strict: bool,
) -> Dict[str, object]:
    """Open a small sample of HDF5 files to validate expected keys.

    Note: this intentionally keeps validation lightweight: just list keys.
    """

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise ImportError("pandas is required for sample validation") from exc

    n = min(int(sample_n), len(records))
    if n <= 0:
        return {
            "sample_n": 0,
            "checked": 0,
            "ok": 0,
            "missing": 0,
            "errors": 0,
            "required_keys": list(required_keys),
        }

    # deterministic-ish sampling with a fixed seed based on current day
    rng = random.Random(0xC0FFEE)
    sample = records if n == len(records) else rng.sample(records, n)

    ok = 0
    missing = 0
    errors = 0
    first_errors: List[str] = []

    req_norm = [k.lstrip("/").lower() for k in required_keys if k]

    for rec in sample:
        try:
            with pd.HDFStore(rec.path, mode="r") as store:
                avail = {k.lstrip("/").lower() for k in store.keys()}
            if all(k in avail for k in req_norm):
                ok += 1
            else:
                missing += 1
                if strict:
                    first_errors.append(f"missing keys: {rec.path}")
        except Exception as e:
            errors += 1
            if len(first_errors) < 10:
                first_errors.append(f"{rec.path}: {type(e).__name__}: {e}")
            if strict:
                # In strict mode, we still continue but report prominently.
                pass

    return {
        "sample_n": n,
        "checked": n,
        "ok": ok,
        "missing": missing,
        "errors": errors,
        "required_keys": list(required_keys),
        "first_errors": first_errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build filesystem index for qcMol .local files")
    ap.add_argument("--data_root", type=str, default="datasets/qcMol/PubChem", help="Directory containing *.local files")
    ap.add_argument("--out_index", type=str, default="qcmol_index.jsonl", help="Output JSONL index path")
    ap.add_argument("--sort", type=str, default="name", choices=["name", "mtime", "size", "none"], help="Sort order")
    ap.add_argument("--sample_validate", type=int, default=0, help="Open N sample files to validate required keys")
    ap.add_argument("--required_keys", type=str, default="XYZ,atom_basic,bond_basic", help="Comma-separated required keys")
    ap.add_argument("--schema_out", type=str, default="qcmol_schema_report.json", help="Optional JSON output for sample validation report")
    ap.add_argument("--strict", type=int, default=0, help="Treat missing keys as important in report")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_index = os.path.abspath(args.out_index)

    t0 = time.time()
    records = list(_iter_local_files(data_root))

    if args.sort == "name":
        records.sort(key=lambda r: os.path.basename(r.path))
    elif args.sort == "mtime":
        records.sort(key=lambda r: r.mtime)
    elif args.sort == "size":
        records.sort(key=lambda r: r.size)

    os.makedirs(os.path.dirname(out_index) or ".", exist_ok=True)
    _write_jsonl(records, out_index)

    print(f"[index] files={len(records)} root={data_root}")
    print(f"[index] wrote: {out_index}")
    print(f"[index] elapsed_sec={time.time() - t0:.2f}")

    if int(args.sample_validate) > 0:
        required_keys = tuple(k.strip() for k in str(args.required_keys).split(",") if k.strip())
        report = _sample_validate(records, int(args.sample_validate), required_keys, strict=bool(args.strict))
        print(
            f"[validate] checked={report.get('checked')} ok={report.get('ok')} missing={report.get('missing')} errors={report.get('errors')}"
        )
        first_errors = report.get("first_errors")
        if isinstance(first_errors, list) and first_errors:
            print("[validate] first_errors:")
            for line in first_errors:
                print("  " + str(line))
        if args.schema_out:
            schema_out = os.path.abspath(args.schema_out)
            os.makedirs(os.path.dirname(schema_out) or ".", exist_ok=True)
            tmp = schema_out + f".tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            os.replace(tmp, schema_out)
            print(f"[validate] wrote: {schema_out}")


if __name__ == "__main__":
    main()
