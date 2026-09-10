"""
bids_demographics_scan.py
--------------------------
Projde BIDS strukturu a vytvoří tabulku s demografickými daty a binárními
příznaky dostupných sekvencí pro každý subject/session.

Výstupní sloupce:
    sub, ses, age, sex,
    has_inv2, has_uni, has_t1map, has_t2map, has_flair, has_t1w

Zdroje demografických dat:
    - MR Exams export (TXT, cp1250, tab-separated)
    - iMed export     (TXT, latin-1, tab-separated)
    Propojení přes patient_id_mapper (RC → paciid)

Použití:
    python bids_demographics_scan.py
    python bids_demographics_scan.py --bids E:\\bids --out missing_check.xlsx
"""

import argparse
import glob
import os

import pandas as pd
from patient_id_mapper.mapper import _load_mapping, _to_int64, _only_digits

# ---------------------------------------------------------------------------
# Konfigurace — přizpůsob názvům sloupců ve svých exportech
# ---------------------------------------------------------------------------

# MR Exams TXT export (cp1250, tab-separated)
MR_PATH   = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\MrExams\ValuesWithHeader_T1_FLAIR_20260304.TXT"
MR_COL_PACIID      = "paciid"
MR_COL_DATE        = "DatumVys"
MR_COL_AGE         = "VekPriVys"
MR_COL_SEX         = "Sex"

# iMed export (latin-1, tab-separated)
IMED_PATH = r"C:\Users\mri-admin\Documents\DATA\iMed_export\IMED export 26_1_2026\imed_export 26_1_2026_ID.txt"

# BIDS adresář
BIDS_DIR  = r"E:\Project_DCE\DCE_PRG_NEW\bids"

# Výstupní soubor
OUT_PATH  = os.path.join(BIDS_DIR, "bids_demographics_sequences.xlsx")

# Glob patterny sekvencí v anat/
SEQUENCES = {
    "has_inv2":  "*inv-2_MP2RAGE*.nii.gz",
    "has_uni":   "*uni_MP2RAGE*.nii.gz",
    "has_t1map": "*t1map_MP2RAGE*.nii.gz",
    "has_t2map": "*t2map*.nii.gz",
    "has_flair": "*FLAIR*.nii.gz",
    "has_t1w":   "*T1w*.nii.gz",
}


# ---------------------------------------------------------------------------
# 1. Načtení demografické tabulky z MR Exams exportu
# ---------------------------------------------------------------------------

def load_mr_demographics(
    mr_path: str,
    col_paciid: str = MR_COL_PACIID,
    col_date: str   = MR_COL_DATE,
    col_age: str    = MR_COL_AGE,
    col_sex: str    = MR_COL_SEX,
) -> pd.DataFrame:
    """
    Načte MR Exams TXT export a vrátí tabulku s paciid, AcquisitionDate, age, sex.

    Poznámka k datu:
      - Pokud je datum ve formátu YYYYMMDD (int), použije se přímo.
      - Pokud je ve formátu DD.MM.YYYY nebo YYYY-MM-DD, převede se automaticky.
    """
    mr = pd.read_csv(
        mr_path,
        sep="\t",
        header=0,
        encoding="cp1250",
        dtype="string",
        low_memory=False,
    )

    needed = [col_paciid, col_date, col_age, col_sex]
    missing = [c for c in needed if c not in mr.columns]
    if missing:
        raise KeyError(
            f"MR Exams export neobsahuje sloupce: {missing}\n"
            f"Dostupné sloupce: {list(mr.columns)}"
        )

    mr = mr[needed].copy()
    mr[col_paciid] = mr[col_paciid].str.strip()

    # Normalizace data → int YYYYMMDD
    mr[col_date] = mr[col_date].str.strip()
    mr["AcquisitionDate"] = _parse_date_to_int(mr[col_date])

    # Věk — desetinná čárka -> tečka, zaokrouhlit na int
    mr["age"] = (
        pd.to_numeric(
            mr[col_age].str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        .round()
        .astype("Int64")
    )

    # Pohlaví — normalizace na M/F
    mr["sex"] = (
        mr[col_sex].str.strip()
        .str.replace("Muž",  "M", regex=False)
        .str.replace("Žena", "F", regex=False)
        .str.replace("muž",  "M", regex=False)
        .str.replace("žena", "F", regex=False)
        .str.replace("Male",   "M", regex=False)
        .str.replace("Female", "F", regex=False)
        .str.replace("1", "M", regex=False)
        .str.replace("2", "F", regex=False)
    )

    # paciid jako Int64
    mr["paciid"] = _to_int64(_only_digits(mr[col_paciid]))

    return mr[["paciid", "AcquisitionDate", "age", "sex"]].dropna(subset=["paciid", "AcquisitionDate"])


def _parse_date_to_int(series: pd.Series) -> pd.Series:
    """
    Převede různé formáty data na int YYYYMMDD.
    Podporuje:
      - YYYYMMDD jako string nebo int ("20240106")
      - DD.MM.YYYY nebo YYYY-MM-DD
      - Excel serial number jako float (41.19, 41936.0, ...)
        Hodnoty < 3000 jsou Excel serial dny od 1900-01-01.
    """
    s = series.copy()

    # Zkus přímý int/string YYYYMMDD (8 číslic)
    direct = pd.to_numeric(s, errors="coerce")

    # Excel serial number: float < 100000, ale hodnoty jsou ~30000-50000 pro roky 1982-2036
    # YYYYMMDD je vždy >= 19000101, takže serial čísla jsou jasně odlišitelná
    excel_mask = direct.notna() & (direct < 100000)
    yyyymmdd_mask = direct.notna() & (direct >= 19000101)

    result = pd.Series(pd.NA, index=s.index, dtype="Int64")

    # Excel serial -> datum
    if excel_mask.any():
        excel_dates = pd.to_datetime(
            direct[excel_mask], unit="D", origin="1899-12-30"
        )
        result[excel_mask] = excel_dates.dt.strftime("%Y%m%d").pipe(
            pd.to_numeric, errors="coerce"
        ).astype("Int64")

    # Přímý YYYYMMDD int
    result[yyyymmdd_mask] = direct[yyyymmdd_mask].astype("Int64")

    # String formáty (DD.MM.YYYY, YYYY-MM-DD, ...)
    string_mask = result.isna() & s.notna()
    if string_mask.any():
        parsed = pd.to_datetime(s[string_mask], dayfirst=True, errors="coerce")
        result[string_mask] = parsed.dt.strftime("%Y%m%d").pipe(
            pd.to_numeric, errors="coerce"
        ).astype("Int64")

    return result


# ---------------------------------------------------------------------------
# 2. Scan BIDS → tabulka sub / ses / sekvence
# ---------------------------------------------------------------------------

def scan_bids(bids_dir: str) -> pd.DataFrame:
    """
    Projde BIDS strukturu a pro každý sub/ses zjistí dostupné sekvence.
    Vrací DataFrame s: sub, ses, paciid (int), AcquisitionDate (int), has_*
    """
    rows = []

    subjects = sorted(
        d for d in os.listdir(bids_dir)
        if os.path.isdir(os.path.join(bids_dir, d)) and d.startswith("sub-")
    )

    for subject in subjects:
        sessions = sorted(
            d for d in os.listdir(os.path.join(bids_dir, subject))
            if os.path.isdir(os.path.join(bids_dir, subject, d)) and d.startswith("ses-")
        )
        for session in sessions:
            anat = os.path.join(bids_dir, subject, session, "anat")

            row = {
                "sub": subject,
                "ses": session,
                "paciid": int(subject[4:]),
                "AcquisitionDate": int(session[4:]),
            }

            if os.path.isdir(anat):
                for flag, pattern in SEQUENCES.items():
                    row[flag] = int(bool(glob.glob(os.path.join(anat, pattern))))
            else:
                for flag in SEQUENCES:
                    row[flag] = 0

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Merge BIDS + demografika
# ---------------------------------------------------------------------------

def build_table(
    bids_dir: str  = BIDS_DIR,
    mr_path: str   = MR_PATH,
    imed_path: str = IMED_PATH,
    mr_col_paciid: str = MR_COL_PACIID,
    mr_col_date: str   = MR_COL_DATE,
    mr_col_age: str    = MR_COL_AGE,
    mr_col_sex: str    = MR_COL_SEX,
) -> pd.DataFrame:

    print("Scanuji BIDS strukturu...")
    bids = scan_bids(bids_dir)
    print(f"  {len(bids)} sessions nalezeno")

    print("Načítám MR Exams demografiku...")
    demog = load_mr_demographics(mr_path, mr_col_paciid, mr_col_date, mr_col_age, mr_col_sex)
    print(f"  {len(demog)} záznamů z MR Exams")

    # Merge přes paciid + AcquisitionDate (integer)
    df = bids.merge(
        demog,
        on=["paciid", "AcquisitionDate"],
        how="left",
    )

    # Integer AcquisitionDate už nepotřebujeme — ses_str ho nahradí
    df = df.drop(columns=["AcquisitionDate"])

    df = df.rename(columns={"age": "PatientAge", "sex": "PatientSex"})

    # PatientID = zero-padded 6-digit string (084686)
    df["PatientID"] = df["paciid"].apply(lambda x: str(x).zfill(6))

    # AcquisitionDate = int (ses-20240512 → 20240512)
    df["AcquisitionDate"] = df["ses"].str[4:].astype(int)

    # DatumVys z ses (ses-20240512 → 12.05.2024)
    df["DatumVys"] = pd.to_datetime(df["ses"].str[4:], format="%Y%m%d").dt.strftime("%d.%m.%Y")

    # paciid jako string bez prefixu
    df["paciid"] = df["paciid"].astype(str)

    cols = ["PatientID", "AcquisitionDate", "paciid", "DatumVys", "PatientAge", "PatientSex"] + list(SEQUENCES.keys())
    df = df[cols]

    # Statistika nenalezených
    missing_demog = df["PatientAge"].isna().sum()
    if missing_demog > 0:
        print(f"  POZOR: {missing_demog} sessions bez demografiky (patient nenalezen v MR Exams)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids",      default=BIDS_DIR,       help="BIDS root directory")
    parser.add_argument("--mr",        default=MR_PATH,        help="MR Exams TXT export")
    parser.add_argument("--imed",      default=IMED_PATH,      help="iMed export TXT")
    parser.add_argument("--out",       default=OUT_PATH,        help="Výstupní soubor (.xlsx nebo .tsv)")
    parser.add_argument("--mr-paciid", default=MR_COL_PACIID,  help="Sloupec paciid v MR Exams")
    parser.add_argument("--mr-date",   default=MR_COL_DATE,    help="Sloupec datum v MR Exams")
    parser.add_argument("--mr-age",    default=MR_COL_AGE,     help="Sloupec věk v MR Exams")
    parser.add_argument("--mr-sex",    default=MR_COL_SEX,     help="Sloupec pohlaví v MR Exams")
    args = parser.parse_args()

    df = build_table(
        bids_dir=args.bids,
        mr_path=args.mr,
        imed_path=args.imed,
        mr_col_paciid=args.mr_paciid,
        mr_col_date=args.mr_date,
        mr_col_age=args.mr_age,
        mr_col_sex=args.mr_sex,
    )

    print(f"\nVýsledná tabulka: {len(df)} řádků, {len(df.columns)} sloupců")
    print(df.head(10).to_string(index=False))

    if args.out.endswith(".xlsx"):
        df.to_excel(args.out, index=False)
    else:
        df.to_csv(args.out, sep="\t", index=False)

    print(f"\nUloženo: {args.out}")


if __name__ == "__main__":
    main()