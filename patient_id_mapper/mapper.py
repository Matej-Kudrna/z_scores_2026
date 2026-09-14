"""
Core mapping functions for patient ID management.
"""

import pandas as pd


def _only_digits(s: pd.Series) -> pd.Series:
    """Keep only 0-9 from strings (preserves NA)."""
    s = pd.Series(s, dtype="string")
    return s.str.replace(r"[^0-9]", "", regex=True)


def _to_int64(s: pd.Series) -> pd.Series:
    """Convert digit-strings to pandas nullable Int64."""
    s = pd.Series(s, dtype="string")
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _norm_rc_string(s: pd.Series) -> pd.Series:
    """RC normalization on the string form (remove '/' and spaces)."""
    s = pd.Series(s, dtype="string")
    return s.str.replace(r"[\/\s]", "", regex=True)


# ----------------------------
# Mapping builder
# ----------------------------
def _load_mapping(
    mr_path=r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\MrExams\ValuesWithHeader_T1_FLAIR_20260304.TXT",
    imed_path=r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\iMed_export\IMED export 26_1_2026\imed_export 26_1_2026_ID.txt",
    mr_rodcislo_col="RodCislo",
    imed_rodcislo_col="Patient Code",
):
    """
    Build mapping with: RC, iMed_ID, paciid as Int64 (nullable).
    All reading and cleaning is done in string space first, then cast to Int64 for output/merge.
    """

    # --- MR side (read as strings)
    mr = pd.read_csv(
        mr_path, sep="\t", header=0, encoding="cp1250",
        dtype={mr_rodcislo_col: "string", "paciid": "string"}
    )
    mr = mr[[mr_rodcislo_col, "paciid"]].dropna().drop_duplicates()

    # --- iMed side (read as strings)
    imed_id_path = imed_path
    imed = pd.read_csv(
        imed_id_path, sep="\t", header=0, encoding="latin-1",
        dtype={imed_rodcislo_col: "string", "Patient ID": "string"}
    )
    imed = imed[[imed_rodcislo_col, "Patient ID"]].dropna().drop_duplicates()

    # --- Normalize RC in string space
    mr[mr_rodcislo_col] = _norm_rc_string(mr[mr_rodcislo_col])
    imed[imed_rodcislo_col] = _norm_rc_string(imed[imed_rodcislo_col])

    # --- Drop temp/test Patient Codes (still in string space)
    imed = imed[~imed[imed_rodcislo_col].str.startswith("BEZISVFN", na=False)]

    # --- Merge via RC (string) first to keep maximum linkage
    map_df = imed.merge(
        mr, left_on=imed_rodcislo_col, right_on=mr_rodcislo_col, how="outer"
    ).rename(columns={"Patient ID": "iMed_ID"})

    # --- Canonical string columns
    map_df["RC"] = map_df[mr_rodcislo_col]
    map_df = map_df[["RC", "iMed_ID", "paciid"]].drop_duplicates()

    # --- Build numeric versions for matching and final output
    map_df["RC_num"] = _to_int64(_only_digits(map_df["RC"]))
    map_df["iMed_ID_num"] = _to_int64(_only_digits(map_df["iMed_ID"]))
    map_df["paciid_num"] = _to_int64(_only_digits(map_df["paciid"]))

    # --- Final mapping returned in Int64 (nullable)
    out = map_df[["RC_num", "iMed_ID_num", "paciid_num"]].drop_duplicates().rename(
        columns={"RC_num": "RC", "iMed_ID_num": "iMed_ID", "paciid_num": "paciid"}
    )
    # All three are Int64 (nullable); NA stays <NA>
    return out


# ----------------------------
# Generic coalescing merge
# ----------------------------
def _coalesce_into(df, left_key, map_df, right_key):
    """
    Merge df with map_df using numeric keys and coalesce RC/iMed_ID/paciid into df as Int64.
    - df[left_key] is read as string but matched numerically (digits only).
    - map_df has RC/iMed_ID/paciid already as Int64 for both merge and output.
    """
    df2 = df.copy()

    # Build numeric left key for merge
    left_num_col = f"__{left_key}_num__"
    df2[left_num_col] = _to_int64(_only_digits(df2[left_key].astype("string")))

    # Prepare right (mapping) numeric column name to merge on
    right_num_col = f"{right_key}"
    merge = df2.merge(
        map_df.add_suffix("_map"),
        left_on=left_num_col,
        right_on=f"{right_num_col}_map",
        how="left"
    )

    # Coalesce into Int64 outputs
    for col in ["RC", "iMed_ID", "paciid"]:
        src = f"{col}_map"
        if col in merge.columns:
            # If df already has that column, try to convert it numerically and prefer existing non-NA
            existing = _to_int64(_only_digits(merge[col].astype("string")))
            from_map = merge[src].astype("Int64")
            merge[col] = existing.where(existing.notna(), from_map)
        else:
            merge[col] = merge[src].astype("Int64")

    # Cleanup temp columns
    drop_cols = [c for c in merge.columns if c.endswith("_map")] + [left_num_col]
    merge = merge.drop(columns=drop_cols, errors="ignore")

    # Ensure final dtypes are Int64
    for col in ["RC", "iMed_ID", "paciid"]:
        merge[col] = merge[col].astype("Int64")

    return merge


def fill_from_RC(df: pd.DataFrame, rc_col: str = "RC", **kwargs):
    """
    Fill iMed_ID and paciid columns based on RC (rodné číslo).
    
    Args:
        df: DataFrame with RC column
        rc_col: Name of the RC column (default: "RC")
        **kwargs: Optional paths to mapping files (mr_path, imed_path)
    
    Returns:
        DataFrame with RC, iMed_ID, and paciid columns filled
    
    Example:
        >>> df = pd.DataFrame({'RC': ['123456/7890', '234567/8901']})
        >>> df_filled = fill_from_RC(df)
    """
    if rc_col not in df.columns:
        raise KeyError(f"Input df must contain '{rc_col}'.")
    df2 = df.copy()
    # Keep original col as string for reading; normalize RC string (slashes/spaces) for better numeric parse
    df2[rc_col] = _norm_rc_string(df2[rc_col].astype("string"))
    map_df = _load_mapping(**kwargs)
    return _coalesce_into(df2, left_key=rc_col, map_df=map_df, right_key="RC")


def fill_from_iMed(df: pd.DataFrame, imed_col: str = "iMed_ID", **kwargs):
    """
    Fill RC and paciid columns based on iMed_ID.
    
    Args:
        df: DataFrame with iMed_ID column
        imed_col: Name of the iMed_ID column (default: "iMed_ID")
        **kwargs: Optional paths to mapping files (mr_path, imed_path)
    
    Returns:
        DataFrame with RC, iMed_ID, and paciid columns filled
    
    Example:
        >>> df = pd.DataFrame({'iMed_ID': ['12345', '23456']})
        >>> df_filled = fill_from_iMed(df)
    """
    if imed_col not in df.columns:
        raise KeyError(f"Input df must contain '{imed_col}'.")
    df2 = df.copy()
    df2[imed_col] = df2[imed_col].astype("string")
    map_df = _load_mapping(**kwargs)
    return _coalesce_into(df2, left_key=imed_col, map_df=map_df, right_key="iMed_ID")


def normalize_paciid(series: pd.Series) -> pd.Series:
    """
    Normalize paciid format.
    
    Removes spaces, converts to uppercase, removes 'sub-' prefix,
    and strips leading zeros.
    
    Args:
        series: Pandas Series with paciid values
    
    Returns:
        Normalized Series with cleaned paciid values
    
    Example:
        >>> s = pd.Series(['sub-00123', 'SUB-456', '  789  '])
        >>> normalize_paciid(s)
        0    123
        1    456
        2    789
        dtype: object
    """
    return (
        series.astype(str)
              .str.strip()                    # remove spaces/tabs
              .str.upper()                    # unify case
              .str.replace(r"^sub-", "", regex=True)  # drop "sub-" if present
              .str.lstrip("0")                # remove leading zeros (00123 → 123)
    )
