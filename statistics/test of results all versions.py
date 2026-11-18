"""
Fast comparison of the data on the first sheet of two Excel files.

Usage:
    python compare_excel.py old.xlsx new.xlsx --key subject_id session
    python compare_excel.py old.xlsx new.xlsx              # no key: compare row by row by position
    python compare_excel.py old.xlsx new.xlsx --key ID --atol 1e-4 --out diff.csv

Speed:
  - reads with the 'calamine' engine (Rust), which is ~10-50x faster than openpyxl
    -> pip install python-calamine   (needs pandas >= 2.2)
  - the comparison is fully vectorised in numpy (no loop over 5.5k columns)
"""
import argparse
import time

import numpy as np
import pandas as pd


def read_csv_fast(path):
    # detect the separator from the first line (, ; or tab)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    sep = max([",", ";", "\t"], key=first.count)
    try:
        return pd.read_csv(path, sep=sep, engine="pyarrow")   # fastest
    except Exception:
        return pd.read_csv(path, sep=sep, low_memory=False)


def read_first_sheet(path):
    t = time.time()
    if path.lower().endswith((".csv", ".tsv", ".txt")):
        df = read_csv_fast(path)
        print(f"  {path}: {df.shape[0]} rows x {df.shape[1]} columns ({time.time() - t:.1f} s)")
        return df
    try:
        df = pd.read_excel(path, sheet_name=0, engine="calamine")
    except (ImportError, ValueError):
        print("  (calamine not available -> openpyxl, will be slower; pip install python-calamine)")
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    print(f"  {path}: {df.shape[0]} rows x {df.shape[1]} columns ({time.time() - t:.1f} s)")
    return df


def compare(f1, f2, key=None, atol=1e-6, rtol=0.0, out="excel_diff.csv"):
    print("Reading...")
    a, b = read_first_sheet(f1), read_first_sheet(f2)
    same = True

    # --- columns ---
    ca, cb = list(a.columns), list(b.columns)
    only_a = [c for c in ca if c not in set(cb)]
    only_b = [c for c in cb if c not in set(ca)]
    if only_a or only_b:
        same = False
        print(f"\nColumns only in file 1 ({len(only_a)}): {only_a[:20]}{' ...' if len(only_a) > 20 else ''}")
        print(f"Columns only in file 2 ({len(only_b)}): {only_b[:20]}{' ...' if len(only_b) > 20 else ''}")
    for name, df in (("1", a), ("2", b)):
        dup = df.columns[df.columns.duplicated()].tolist()
        if dup:
            print(f"WARNING: duplicate column names in file {name}: {dup[:10]}")

    # --- rows ---
    if key:
        for name, df in (("1", a), ("2", b)):
            d = df.duplicated(subset=key)
            if d.any():
                print(f"WARNING: duplicate key in file {name}: {d.sum()}x, e.g. {df.loc[d, key].head(5).values.tolist()}")
        # duplicate keys: add a sequence number (_dup) so duplicates get paired
        # in order (1st with 1st, 2nd with 2nd) instead of the alignment breaking
        def keyed(df):
            k = df[key].copy()
            k["_dup"] = df.groupby(key, dropna=False).cumcount()
            out = df.drop(columns=key)
            out.index = pd.MultiIndex.from_frame(k)
            return out
        a, b = keyed(a), keyed(b)
        ra, rb = a.index.difference(b.index), b.index.difference(a.index)
        if len(ra) or len(rb):
            same = False
            print(f"\nRows only in file 1 ({len(ra)}): {list(ra[:20])}")
            print(f"Rows only in file 2 ({len(rb)}): {list(rb[:20])}")
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
    else:
        if len(a) != len(b):
            same = False
            print(f"\nDifferent number of rows: {len(a)} vs {len(b)} -> comparing only the first {min(len(a), len(b))}")
        n = min(len(a), len(b))
        a, b = a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)

    cols = [c for c in a.columns if c in set(b.columns)]
    a, b = a[cols], b[cols]

    # --- split into numeric / other columns ---
    num_mask = np.array([pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]) for c in cols])
    num_cols = [c for c, m in zip(cols, num_mask) if m]
    obj_cols = [c for c, m in zip(cols, num_mask) if not m]

    diffs = []
    t = time.time()

    if num_cols:
        A = a[num_cols].to_numpy(dtype=float)
        B = b[num_cols].to_numpy(dtype=float)
        eq = np.isclose(A, B, atol=atol, rtol=rtol, equal_nan=True)
        r, c = np.nonzero(~eq)
        if len(r):
            diffs.append(pd.DataFrame({
                "row": a.index[r], "column": np.array(num_cols, dtype=object)[c],
                "file1": A[r, c], "file2": B[r, c], "abs_diff": np.abs(A[r, c] - B[r, c]),
            }))

    if obj_cols:
        A = a[obj_cols].to_numpy(dtype=object)
        B = b[obj_cols].to_numpy(dtype=object)
        na_a, na_b = pd.isna(A), pd.isna(B)
        # compare as strings (5 vs '5', numbers in text columns...), NaN == NaN
        eq = (A.astype(str) == B.astype(str)) | (na_a & na_b)
        eq &= ~(na_a ^ na_b)
        r, c = np.nonzero(~eq)
        if len(r):
            diffs.append(pd.DataFrame({
                "row": a.index[r], "column": np.array(obj_cols, dtype=object)[c],
                "file1": A[r, c], "file2": B[r, c], "abs_diff": np.nan,
            }))

    print(f"\nCompared {len(a)} rows x {len(cols)} columns "
          f"({len(num_cols)} numeric, {len(obj_cols)} text) in {time.time() - t:.2f} s")

    if diffs:
        same = False
        d = pd.concat(diffs, ignore_index=True)
        print(f"\n{len(d)} differing cells in {d['column'].nunique()} columns and {d['row'].nunique()} rows")
        summary = (d.groupby("column")
                     .agg(n_diff=("row", "size"), max_abs_diff=("abs_diff", "max"))
                     .sort_values("n_diff", ascending=False))
        print("\nColumns with the most differences:")
        print(summary.head(20).to_string())
        d.to_csv(out, index=False)
        summary.to_csv(out.replace(".csv", "_by_column.csv"))
        print(f"\nAll differences saved to: {out} (+ {out.replace('.csv', '_by_column.csv')})")
    else:
        print("No differing values in the common rows/columns.")

    print("\n✅ DATA ARE IDENTICAL" if same else "\n❌ DATA DIFFER")
    return same


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--key", nargs="+", default=None, help="key column(s), e.g. --key subject_id session")
    p.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance for numbers")
    p.add_argument("--rtol", type=float, default=0.0, help="relative tolerance for numbers")
    p.add_argument("--out", default="excel_diff.csv")
    args = p.parse_args()
    compare(args.file1, args.file2, key=args.key, atol=args.atol, rtol=args.rtol, out=args.out)