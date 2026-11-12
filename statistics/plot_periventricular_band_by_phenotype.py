"""PDF report periventrikulárního gradientu (T1) rozděleného podle center.

Použití: uprav nastavení v bloku `if __name__ == "__main__":` dole a spusť skript,
nebo z jiného skriptu:
    from perivent_gradient_report import run_report
    run_report("data.csv", centre_from_id="ID")

Vstup: tabulka ve wide formátu (1 řádek = 1 subjekt/sezení) se sloupci
    map_perivent_<metrika>_t1_band_{1..48}_{lesions_in_band,
    nasc_in_band, nawm_in_band, nawmsc_in_band, whole_band}
a sloupcem s názvem centra.

Report obsahuje:
  1. přehled: vstup, počty subjektů na centrum, přesné názvy použitých sloupců
  2. souhrn – všechna centra v jednom grafu jen jako křivky (medián + pás IQR),
     zvlášť absolutní objem a podíl z objemu pásu (in_band / whole_band × 100);
     ke každému grafu tabulka medián [IQR] a Kruskal–Wallis mezi centry po pásech
     s korekcí FDR (Benjamini–Hochberg)
  3. oddíl pro každé centrum: boxploty po pásech + křivka mediánu a pás IQR
"""
from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy.stats import kruskal

METRIC = "volume_absolute_greater_2"   # přepíše run_report(metric=...)
PREFIX = f"map_perivent_{METRIC}"
METRIC_LABELS = {
    "volume_absolute_greater_2": "objem nad prahem |x| > 2 (voxely)",
    "relative_volume_absolute_greater_2": "podíl voxelů nad prahem |x| > 2",
    "average": "průměr mapy v ROI",
    "average_absolute": "průměr |mapy| v ROI",
    "median": "medián mapy v ROI",
    "median_absolute": "medián |mapy| v ROI",
    "standard_deviation": "SD mapy v ROI",
}
MODALITY = "t1"
N_BANDS = 48
TISSUES = {
    "lesions_in_band": "Léze v pásu",
    "nawm_in_band": "NAWM v pásu",
    "nawmsc_in_band": "NAWMSC v pásu",
    "nasc_in_band": "NASC v pásu",
    "whole_band": "Celý pás (whole_band)",
}
TABLE_BANDS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48]
# Kategorická paleta v pevném pořadí (centrum má vždy stejnou barvu)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CENTRE_CANDIDATES = ["centre", "center", "centrum", "site", "Centre", "Center", "Site"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def col_name(band: int, tissue: str, modality: str = MODALITY) -> str:
    return f"{PREFIX}_{modality}_band_{band}_{tissue}"


def col_pattern(tissue: str, modality: str = MODALITY) -> str:
    return f"{PREFIX}_{modality}_band_{{1..{N_BANDS}}}_{tissue}"


# --------------------------------------------------------------------------- data
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    sep = ";" if path.read_text(errors="ignore").split("\n", 1)[0].count(";") > 0 else ","
    return pd.read_csv(path, sep=sep)


def find_centre_col(df: pd.DataFrame, given: str | None) -> str:
    if given:
        if given not in df.columns:
            raise SystemExit(f"Sloupec centra '{given}' v datech není.")
        return given
    for c in CENTRE_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit("Nenašel jsem sloupec s centrem, zadej ho přes --centre-col.")


def to_long(df: pd.DataFrame, centre_col: str) -> tuple[pd.DataFrame, dict]:
    """Wide → long (row_id, centre, band, tissue, value); vrací i přehled sloupců."""
    found: dict[str, dict] = {}
    parts = []
    for tissue in TISSUES:
        cols = {b: col_name(b, tissue) for b in range(1, N_BANDS + 1)}
        present = {b: c for b, c in cols.items() if c in df.columns}
        empty = [b for b, c in present.items() if df[c].isna().all()]
        if empty:
            print(f"Pozn.: {tissue}: {len(empty)}/{len(present)} sloupců je úplně prázdných "
                  f"(pásy {compress(empty)}).")
        constant = [b for b, c in present.items() if b not in empty and df[c].nunique(dropna=True) <= 1]
        found[tissue] = {"present": sorted(present), "missing": sorted(set(cols) - set(present)),
                         "constant": constant, "empty": empty}
        if constant and len(df) > 1:
            print(f"VAROVÁNÍ: {tissue}: {len(constant)}/{len(present)} sloupců má u všech "
                  f"řádků stejnou hodnotu (pásy {compress(constant)}) – zkontroluj vstupní data.")
        used = [b for b in present if b not in empty and b not in constant]
        found[tissue]["used"] = used
        for b in used:
            c = present[b]
            parts.append(pd.DataFrame({
                "row_id": df.index, "centre": df[centre_col].astype("string"),
                "band": b, "tissue": tissue,
                "value": pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce"),
            }))
    if not parts:
        raise SystemExit(f"V datech nejsou žádné použitelné sloupce {PREFIX}_{MODALITY}_band_* "
                         "(chybí, jsou prázdné nebo u všech řádků stejné).")
    long = pd.concat(parts, ignore_index=True)
    long = long[long.centre.notna()]
    return long, found


def add_relative(long: pd.DataFrame) -> pd.DataFrame:
    whole = long[long.tissue == "whole_band"][["row_id", "band", "value"]].rename(
        columns={"value": "whole"})
    rel = long[long.tissue != "whole_band"].merge(whole, on=["row_id", "band"], how="left")
    rel["value"] = np.where(rel.whole > 0, 100 * rel.value / rel.whole, np.nan)
    return rel.drop(columns="whole")


def band_stats(sub: pd.DataFrame) -> pd.DataFrame:
    g = sub.dropna(subset=["value"]).groupby(["centre", "band"]).value
    return pd.DataFrame({
        "n": g.size(), "q1": g.quantile(.25), "median": g.median(), "q3": g.quantile(.75),
    }).reset_index()


def kruskal_by_band(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, d in sub.dropna(subset=["value"]).groupby("band"):
        groups = [g.values for _, g in d.groupby("centre").value if len(g) >= 3]
        p = np.nan
        if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
            try:
                p = kruskal(*groups).pvalue
            except ValueError:
                p = np.nan
        rows.append({"band": b, "p": p})
    res = pd.DataFrame(rows)
    res["q"] = bh_fdr(res.p.values)
    return res


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size:
        order = np.argsort(pv)
        ranked = pv[order] * pv.size / np.arange(1, pv.size + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(pv)
        out[order] = np.minimum(ranked, 1)
        q[ok] = out
    return q


# --------------------------------------------------------------------------- figures
def _axes_common(fig, ax, title: str, subtitle: str, ylabel: str, columns_text: str, log: bool):
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0.4, N_BANDS + 0.6)
    ax.set_xticks([1, *range(4, N_BANDS + 1, 4)])
    ax.set_xlabel("pás (band) – vzdálenost od komor →")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.07, 0.945, title, fontsize=14, color=INK)
    fig.text(0.07, 0.915, textwrap.fill(subtitle, 150), fontsize=8.5, color=MUTED, va="top")
    fig.text(0.07, 0.04, columns_text, fontsize=8, color=MUTED, family="monospace",
             va="bottom")


def overview_figure(sub: pd.DataFrame, centres: list[str], colors: dict, title: str,
                    ylabel: str, columns_text: str, kw: pd.DataFrame, log: bool):
    """Všechna centra přes sebe: jen křivka mediánu a pás IQR, bez boxplotů."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 na šířku
    ax = fig.add_axes([0.07, 0.24, 0.9, 0.62])
    stats = band_stats(sub if not log else sub[sub.value > 0])
    for c in centres:
        s = stats[stats.centre == c].sort_values("band")
        if s.empty:
            continue
        ax.fill_between(s.band, s.q1, s.q3, color=colors[c], alpha=0.14, linewidth=0)
        ax.plot(s.band, s["median"], color=colors[c], linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", marker="o", markersize=2.5)
    if not log and not stats.empty:
        ax.set_ylim(min(0, stats.q1.min()), stats.q3.max() * 1.08)
    sig = kw[kw.q < 0.05].band.values if len(centres) > 1 else []
    ymax = ax.get_ylim()[1]
    for b in sig:
        ax.text(b, ymax, "*", ha="center", va="bottom", fontsize=10, color=INK)
    n_per = sub.dropna(subset=["value"]).groupby("centre").row_id.nunique()
    handles = [Patch(facecolor=colors[c] + "33", edgecolor=colors[c],
                     label=f"{c} (n = {n_per.get(c, 0)})") for c in centres]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=min(len(centres), 4))
    _axes_common(fig, ax, title,
                 "Všechna centra přes sebe: čára = medián, stínování = IQR (Q1–Q3). "
                 "* = rozdíl mezi centry v daném pásu (Kruskal–Wallis, FDR q < 0,05).",
                 ylabel, columns_text, log)
    return fig


def centre_figure(sub: pd.DataFrame, centre: str, color: str, title: str, ylabel: str,
                  columns_text: str, log: bool, fliers: bool = False):
    """Jedno centrum: boxplot v každém pásu + křivka mediánu a pás IQR."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.07, 0.24, 0.9, 0.62])
    d = sub[sub.centre == centre].dropna(subset=["value"])
    if log:
        d = d[d.value > 0]
    bands = sorted(d.band.unique())
    data = [d.loc[d.band == b, "value"].to_numpy() for b in bands]
    if data:
        bp = ax.boxplot(
            data, positions=bands, widths=0.6, patch_artist=True, showfliers=fliers,
            manage_ticks=False,
            boxprops={"facecolor": color + "26", "edgecolor": color, "linewidth": 0.7},
            medianprops={"color": color, "linewidth": 1},
            whiskerprops={"color": color, "linewidth": 0.6},
            capprops={"color": color, "linewidth": 0.6},
            flierprops={"marker": ".", "markersize": 2, "markerfacecolor": color,
                        "markeredgecolor": "none", "alpha": 0.5},
        )
        s = band_stats(d).sort_values("band")
        ax.fill_between(s.band, s.q1, s.q3, color=color, alpha=0.12, linewidth=0)
        ax.plot(s.band, s["median"], color=color, linewidth=2, solid_capstyle="round",
                solid_joinstyle="round")
        whisk = [v for w in bp["whiskers"] for v in w.get_ydata()]
        if whisk and not fliers:
            lo, hi = np.nanmin(whisk), np.nanmax(whisk)
            ax.set_ylim((lo / 1.3, hi * 1.3) if log else (min(0, lo), hi * 1.06))
    n = d.row_id.nunique()
    ax.legend(handles=[Patch(facecolor=color + "26", edgecolor=color,
                             label=f"{centre} (n = {n})")],
              loc="upper right", frameon=False)
    _axes_common(fig, ax, title,
                 "Boxplot v každém pásu (whiskery 1,5×IQR); čára = medián, stínování = IQR."
                 + ("" if fliers else " Odlehlé hodnoty nejsou zobrazeny."),
                 ylabel, columns_text, log)
    return fig


def section_page(title: str, lines: list[str]):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.55, title, fontsize=22, color=INK)
    fig.text(0.06, 0.49, "\n".join(lines), fontsize=10, color=MUTED, va="top", linespacing=1.6)
    return fig


def text_page(lines: list[str], title: str):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.93, title, fontsize=15, color=INK)
    fig.text(0.06, 0.88, "\n".join(lines), fontsize=8.5, color=INK, va="top",
             family="monospace", linespacing=1.45)
    return fig


def table_page(sub: pd.DataFrame, centres: list[str], title: str, unit: str, kw: pd.DataFrame,
               columns_text: str):
    st = band_stats(sub)
    rows, cell = [], []
    for c in centres:
        s = st[st.centre == c].set_index("band")
        cell.append([fmt_cell(s, b) for b in TABLE_BANDS])
        rows.append(c)
    kwi = kw.set_index("band")
    cell.append([fmt_p(kwi.q.get(b, np.nan)) for b in TABLE_BANDS])
    rows.append("KW q (FDR)")
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.93, title, fontsize=14, color=INK)
    fig.text(0.06, 0.895, f"Medián [Q1–Q3] ({unit}); n = počet subjektů s hodnotou. "
             "Poslední řádek: Kruskal–Wallis mezi centry, FDR (Benjamini–Hochberg) přes 48 pásů.",
             fontsize=8.5, color=MUTED)
    ax = fig.add_axes([0.06, 0.2, 0.9, 0.65])
    ax.axis("off")
    tb = ax.table(cellText=cell, rowLabels=rows, colLabels=[f"band {b}" for b in TABLE_BANDS],
                  loc="upper left", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.8)
    tb.scale(1, 2.6)
    for (r, _), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if r == 0:
            c.set_text_props(color=MUTED)
    fig.text(0.06, 0.04, columns_text, fontsize=8, color=MUTED, family="monospace",
             va="bottom")
    return fig


def fmt_cell(s: pd.DataFrame, b: int) -> str:
    if b not in s.index:
        return "–"
    r = s.loc[b]
    return f"{fmt_num(r['median'])}\n[{fmt_num(r.q1)}–{fmt_num(r.q3)}]\nn={int(r.n)}"


def fmt_num(v: float) -> str:
    if pd.isna(v):
        return "–"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def fmt_p(q: float) -> str:
    return "–" if pd.isna(q) else ("<0.001" if q < 0.001 else f"{q:.3f}")


# --------------------------------------------------------------------------- main
def build_report(df: pd.DataFrame, out: Path, centre_col: str, log: bool,
                 source_name: str, centre_order: list[str] | None = None,
                 fliers: bool = False, centre_label: str | None = None) -> None:
    long, found = to_long(df, centre_col)
    centres = centre_order or sorted(long.centre.dropna().unique())
    if len(centres) > len(PALETTE):
        raise SystemExit(f"Víc než {len(PALETTE)} center – sluč menší centra do 'Other'.")
    colors = {c: PALETTE[i] for i, c in enumerate(centres)}
    is_volume = METRIC.startswith("volume")
    rel = add_relative(long) if is_volume and found["whole_band"]["used"] else None
    n_rows = long.groupby("centre").row_id.nunique()

    intro = [
        f"Vstup:           {source_name}",
        f"Vytvořeno:       {date.today().isoformat()}",
        f"Mapa:            {MODALITY.upper()} (sloupce {PREFIX}_{MODALITY}_band_*)",
        f"Metrika:         {METRIC} – {METRIC_LABELS.get(METRIC, METRIC)}",
        f"Centrum:         {centre_label or centre_col}",
        (f"Osa y:           {'logaritmická' if log else 'lineární'}"
         f"{'' if fliers else '; u boxplotů oříznutá na rozsah whiskerů'}"),
        "",
        "Počet řádků (subjektů/sezení) na centrum:",
        *[f"  {c:<20s} {n_rows.get(c, 0)}" for c in centres],
        "",
        "Vykreslené tkáně (jen pásy, kde se hodnoty mezi subjekty liší):",
    ]
    skipped = []
    for tissue, label in TISSUES.items():
        f = found[tissue]
        if f["used"]:
            intro.append(f"  {label:<24s} {col_pattern(tissue)}")
            intro.append(f"  {'':<24s} pásy {compress(f['used'])}")
        notes = [f"{what} {compress(f[k])}" for what, k in
                 (("prázdné", "empty"), ("konstantní", "constant"), ("chybí", "missing")) if f[k]]
        if notes:
            skipped.append(f"  {label:<24s} " + "; ".join(notes))
    if skipped:
        intro += ["", "Vynechané pásy (bez informace – nekreslí se ani netestují):", *skipped]
    intro += [
        "",
        *(["Relativní grafy: 100 × <tkáň>_in_band / whole_band (stejný pás a subjekt)."] if rel is not None
          else []),
        "Chybějící a nečíselné hodnoty se vynechávají pro daný pás; desetinná čárka se převádí.",
        "Statistika: Kruskal–Wallis mezi centry pro každý pás (centra s n ≥ 3),",
        "korekce FDR Benjamini–Hochberg přes vykreslené pásy dané tkáně.",
    ]

    ylabel = METRIC_LABELS.get(METRIC, METRIC)
    variants = []  # (tissue, nadpis, popisek osy y, data, popis sloupců)
    for tissue, label in TISSUES.items():
        if not found[tissue]["used"]:
            continue
        variants.append((tissue, f"{label} – absolutní objem" if is_volume else f"{label} – {ylabel}",
                         ylabel, long[long.tissue == tissue], f"Sloupce: {col_pattern(tissue)}"))
        if rel is not None and tissue != "whole_band":
            r = rel[(rel.tissue == tissue)].dropna(subset=["value"])
            if r.groupby("band").value.nunique().gt(1).any():
                variants.append((tissue, f"{label} – podíl z objemu pásu", "% objemu pásu", r,
                                 f"Čitatel:    {col_pattern(tissue)}\n"
                                 f"Jmenovatel: {col_pattern('whole_band')}"))
    if not variants:
        raise SystemExit("Nic k vykreslení – všechny sloupce jsou prázdné nebo konstantní.")

    with PdfPages(out) as pdf:
        pdf.savefig(text_page(intro, f"Periventrikulární gradient – T1, {METRIC}, podle center"))
        plt.close("all")

        pdf.savefig(section_page("1  Souhrn – všechna centra", [
            "Křivky mediánu a pásy IQR všech center v jednom grafu, bez boxplotů.",
            f"Metrika: {METRIC} – {METRIC_LABELS.get(METRIC, METRIC)}",
            "Za každým grafem tabulka medián [IQR] ve vybraných pásech a test mezi centry.",
            "Centra: " + ", ".join(centres)]))
        for _tissue, title, ylabel, sub, cols in variants:
            kw = kruskal_by_band(sub)
            pdf.savefig(overview_figure(sub, centres, colors, f"{title} (T1) – všechna centra",
                                        ylabel, cols, kw, log))
            pdf.savefig(table_page(sub, centres, f"{title} (T1) – tabulka", ylabel, kw, cols))
            plt.close("all")

        for i, c in enumerate(centres, start=2):
            pdf.savefig(section_page(f"{i}  Centrum {c}", [
                f"Počet řádků (subjektů/sezení): {n_rows.get(c, 0)}",
                "Boxploty po pásech s křivkou mediánu a pásem IQR."]))
            for _tissue, title, ylabel, sub, cols in variants:
                if sub[(sub.centre == c)].value.notna().sum() == 0:
                    continue
                pdf.savefig(centre_figure(sub, c, colors[c], f"{c}: {title} (T1)", ylabel,
                                          cols, log, fliers))
                plt.close("all")
        meta = pdf.infodict()
        meta["Title"] = "Periventricular gradient T1 by centre"


def compress(bands: list[int]) -> str:
    out, start = [], None
    for i, b in enumerate(bands):
        if start is None:
            start = b
        if i == len(bands) - 1 or bands[i + 1] != b + 1:
            out.append(f"{start}" if start == b else f"{start}–{b}")
            start = None
    return ", ".join(out)


def run_report(
    input_path: str | Path,
    output_path: str | Path | None = None,
    centre_col: str | None = None,
    centre_from_id: str | None = None,
    centres: list[str] | None = None,
    log: bool = False,
    fliers: bool = False,
    metric: str = "median",
) -> Path:
    """Vytvoří PDF report. Volatelné přímo z Pythonu (skript, notebook).

    input_path      CSV nebo XLSX ve wide formátu
    output_path     výstupní PDF; None = vedle vstupu jako <název>_perivent_T1_report.pdf
    centre_col      sloupec s centrem (None = autodetekce centre/center/centrum/site)
    centre_from_id  sloupec s ID; centrum = znaky před první '-' (např. 'PRG-001' → 'PRG')
    centres         pořadí center = pořadí barev, např. ['PRG', 'CPH']; None = abecedně
    log             logaritmická osa y
    fliers          zobrazit v boxplotech odlehlé hodnoty
    metric          která metrika z perivent tabulky: median, average, median_absolute,
                    average_absolute, standard_deviation, volume_absolute_greater_2,
                    relative_volume_absolute_greater_2
    """
    global METRIC, PREFIX
    METRIC, PREFIX = metric, f"map_perivent_{metric}"
    input_path = Path(input_path)
    output_path = (Path(output_path) if output_path else
                   input_path.with_name(f"{input_path.stem}_perivent_T1_{metric}_report.pdf"))
    df = read_table(input_path)
    label = None
    if centre_from_id:
        if centre_from_id not in df.columns:
            raise SystemExit(f"Sloupec '{centre_from_id}' v datech není. "
                             f"Dostupné: {', '.join(map(str, df.columns[:10]))}…")
        ids = df[centre_from_id].astype("string").str.strip()
        no_dash = ids.notna() & ~ids.str.contains("-", regex=False)
        if no_dash.any():
            print(f"Varování: {int(no_dash.sum())} ID bez '-' (např. {ids[no_dash].iloc[0]}),"
                  " tyto řádky nemají centrum a vynechají se.")
        df = df.copy()
        df["_centre"] = ids.where(~no_dash).str.split("-", n=1).str[0].str.strip()
        centre_col = "_centre"
        label = f"znaky před '-' ve sloupci {centre_from_id}"
    else:
        centre_col = find_centre_col(df, centre_col)
    print("Centra:", df[centre_col].value_counts(dropna=False).to_dict())
    build_report(df, output_path, centre_col, log, input_path.name, centres, fliers, label)
    print(f"Hotovo: {output_path}")
    return output_path


if __name__ == "__main__":
    # ---- nastavení ----------------------------------------------------------
    INPUT = r"E:\Project_DCE\DATA_for_merge\combine_results_from_centers_20261030_1508.csv"
    OUTPUT = None            # None = vedle CSV jako <název>_perivent_T1_<metrika>_report.pdf
    METRIC_NAME = "standard_deviation"   # median | average | median_absolute | standard_deviation | volume_absolute_greater_2 ...
    CENTRE_FROM_ID = "Phenotype"    # centrum = znaky před '-' ve sloupci ID
    CENTRES = None           # pořadí/barvy center, např. ["PRG", "CPH", "LAU"]
    LOG = False              # True = logaritmická osa y
    FLIERS = False           # True = zobrazit odlehlé hodnoty
    # ------------------------------------------------------------------------
    run_report(INPUT, OUTPUT, centre_from_id=CENTRE_FROM_ID, centres=CENTRES,
               log=LOG, fliers=FLIERS, metric=METRIC_NAME)