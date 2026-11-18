"""Periventricular gradient report (T1 map) - PDF, grouped by centre or phenotype.

Usage: edit the settings in the `if __name__ == "__main__":` block at the bottom and run
the script, or call it from another script:
    from perivent_gradient_report import run_report
    run_report("data.csv", group_from_id="ID", group_noun="centre")
    run_report("data.csv", group_col="Phenotype", group_noun="phenotype")

Input: wide table (one row = one subject/session) with columns
    map_perivent_<metric>_t1_band_{1..48}_{lesions_in_band, nasc_in_band, nawm_in_band,
                                           nawmsc_in_band, whole_band}

The report contains
  1. overview page: input, metric, inferred y-axis units, group sizes, exact column names,
     and which bands were skipped (missing, empty, or identical in every row)
  2. data plausibility check: subject-level T1 per group, subjects with values far
     from the cohort (possible unit scaling or fitting problems); also saved as CSV
  3. overview: all groups in one plot, median curve + IQR band only (no boxplots), with a
     table of median [IQR] and a Kruskal-Wallis test between groups per band (BH-FDR)
  4. one section per group: boxplots per band + median curve and IQR band
Only informative bands and tissues are drawn.
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

MODALITY = "t1"
N_BANDS = 30
BAND_RANGE = (1, N_BANDS)  # bands actually analysed; set via run_report(band_min=, band_max=)
TISSUES = {
    "nawmsc_in_band": "NAWMSC in band",
    "nawm_in_band": "NAWM in band",
    "nasc_in_band": "NASC in band",
    "lesions_in_band": "Lesions in band",
    "whole_band": "Whole band",
}
METRICS = {
    "median": "median of the map within the ROI",
    "average": "mean of the map within the ROI",
    "median_absolute": "median of |map| within the ROI",
    "average_absolute": "mean of |map| within the ROI",
    "standard_deviation": "standard deviation of the map within the ROI",
    "volume_absolute_greater_2": "number of ROI voxels with |value| > 2",
    "relative_volume_absolute_greater_2": "fraction of ROI voxels with |value| > 2",
}
STAT_NAME = {
    "median": "Median", "average": "Mean", "median_absolute": "Median |value|",
    "average_absolute": "Mean |value|", "standard_deviation": "Within-ROI SD",
}
# Plausible T1 at 3T (ms): WM ~700-1200, deep GM ~1100-1600, cortex ~1300-1900, CSF ~3500-4500
T1_MS_MAX_PLAUSIBLE = 5000
# Fixed-order categorical palette (a group keeps its colour throughout the report)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
A4 = (11.69, 8.27)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------- helpers
def col_name(metric: str, band: int, tissue: str) -> str:
    return f"map_perivent_{metric}_{MODALITY}_band_{band}_{tissue}"


def col_pattern(metric: str, tissue: str) -> str:
    lo, hi = BAND_RANGE
    return f"map_perivent_{metric}_{MODALITY}_band_{{{lo}..{hi}}}_{tissue}"


def compress(bands: list[int]) -> str:
    """[1,2,3,7] -> '1-3, 7'"""
    out, start = [], None
    bands = sorted(bands)
    for i, b in enumerate(bands):
        if start is None:
            start = b
        if i == len(bands) - 1 or bands[i + 1] != b + 1:
            out.append(f"{start}" if start == b else f"{start}-{b}")
            start = None
    return ", ".join(out)


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False), errors="coerce")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = max([",", ";", "\t"], key=head.count)
    return pd.read_csv(path, sep=sep, low_memory=False)


# --------------------------------------------------------------------------- data
def to_long(df: pd.DataFrame, group_col: str, metric: str) -> tuple[pd.DataFrame, dict]:
    """Wide -> long (row_id, group, band, tissue, value), informative bands only."""
    found: dict[str, dict] = {}
    parts = []
    for tissue in TISSUES:
        cols = {b: col_name(metric, b, tissue) for b in range(BAND_RANGE[0], BAND_RANGE[1] + 1)}
        present = {b: c for b, c in cols.items() if c in df.columns}
        vals = {b: to_num(df[c]) for b, c in present.items()}
        empty = [b for b, v in vals.items() if v.isna().all()]
        constant = [b for b, v in vals.items()
                    if b not in empty and len(df) > 1 and v.nunique(dropna=True) <= 1]
        used = [b for b in present if b not in empty and b not in constant]
        found[tissue] = {"present": sorted(present), "missing": sorted(set(cols) - set(present)),
                         "empty": empty, "constant": constant, "used": used}
        if constant:
            print(f"WARNING: {tissue}: {len(constant)}/{len(present)} columns have the same value "
                  f"in every row (bands {compress(constant)}) - skipped, check the input data.")
        for b in used:
            parts.append(pd.DataFrame({
                "row_id": df.index, "group": df[group_col].astype("string"),
                "band": b, "tissue": tissue, "value": vals[b],
            }))
    if not parts:
        raise SystemExit(f"No usable map_perivent_{metric}_{MODALITY}_band_* columns "
                         "(missing, empty, or identical in every row).")
    long = pd.concat(parts, ignore_index=True)
    return long[long.group.notna()], found


def add_relative(long: pd.DataFrame) -> pd.DataFrame:
    whole = long[long.tissue == "whole_band"][["row_id", "band", "value"]].rename(
        columns={"value": "whole"})
    rel = long[long.tissue != "whole_band"].merge(whole, on=["row_id", "band"], how="inner")
    rel["value"] = np.where(rel.whole > 0, 100 * rel.value / rel.whole, np.nan)
    return rel.drop(columns="whole")


def band_stats(sub: pd.DataFrame) -> pd.DataFrame:
    g = sub.dropna(subset=["value"]).groupby(["group", "band"]).value
    return pd.DataFrame({"n": g.size(), "q1": g.quantile(.25), "median": g.median(),
                         "q3": g.quantile(.75)}).reset_index()


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


def kruskal_by_band(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, d in sub.dropna(subset=["value"]).groupby("band"):
        samples = [g.to_numpy() for _, g in d.groupby("group").value if len(g) >= 3]
        p = np.nan
        if len(samples) >= 2 and np.ptp(np.concatenate(samples)) > 0:
            try:
                p = kruskal(*samples).pvalue
            except ValueError:
                p = np.nan
        rows.append({"band": b, "p": p})
    res = pd.DataFrame(rows, columns=["band", "p"])
    res["q"] = bh_fdr(res.p.to_numpy())
    return res


# --------------------------------------------------------------------------- units
def infer_axis(metric: str, values: pd.Series) -> tuple[str, str]:
    """Infer the y-axis label from the metric and the bulk of the values.

    Returns (axis label, one-line explanation of how it was inferred).
    """
    if metric == "volume_absolute_greater_2":
        return ("Volume with |value| > 2 (voxels)",
                "Voxel count of the ROI in the band (multiply by voxel volume for mm³). "
                "Values of 10³-10⁴ are band sizes, not T1 times.")
    if metric == "relative_volume_absolute_greater_2":
        return "Fraction of ROI with |value| > 2", "Fraction between 0 and 1."
    v = pd.Series(values, dtype=float).dropna()
    stat = STAT_NAME.get(metric, metric)
    if v.empty:
        return stat, "No values."
    med = float(v.median())
    is_sd = metric == "standard_deviation"
    if (v < 0).mean() > 0.05 and abs(med) < 10:
        return f"{stat} (z-score)", (f"Negative values and bulk median {med:.2f}: "
                                     "a normative z-score map, not a relaxation time.")
    if (50 if is_sd else 200) <= med <= 6000:
        return (f"{stat} of T1 relaxation time (ms)",
                f"Bulk median {med:.0f}: T1 in milliseconds "
                "(3T reference: WM ~700-1200 ms, deep GM ~1100-1600, cortex ~1300-1900, CSF ~4000).")
    if (0.05 if is_sd else 0.2) <= med <= 6:
        return (f"{stat} (T1 in s or R1 in 1/s - check)",
                f"Bulk median {med:.2f}: T1 in seconds or R1 = 1/T1 in 1/s.")
    return (f"{stat} (unknown units)",
            f"Bulk median {med:.3g} does not match T1 in ms or s - check which map was used.")


def plausibility(long: pd.DataFrame, metric: str, ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Subject-level typical value (median over bands of the most WM-like tissue).

    Flags subjects at < 0.5x or > 2x the cohort median, or above 5000 ms anywhere.
    """
    for tissue in ("nawmsc_in_band", "nawm_in_band", "whole_band"):
        d = long[long.tissue == tissue]
        if d.value.notna().any():
            break
    subj = d.groupby(["row_id", "group"]).value.median().rename("typical").reset_index()
    peak = long.groupby("row_id").value.max().rename("max_any")
    subj = subj.join(peak, on="row_id")
    subj.insert(0, "ID", ids.reindex(subj.row_id).to_numpy())
    cohort = subj.typical.median()
    subj["ratio_to_cohort"] = subj.typical / cohort
    reasons = []
    for _, r in subj.iterrows():
        why = []
        if r.ratio_to_cohort > 2:
            why.append(f"{r.ratio_to_cohort:.1f}x cohort median")
        elif r.ratio_to_cohort < 0.5:
            why.append(f"{r.ratio_to_cohort:.2f}x cohort median")
        if metric in ("median", "average", "median_absolute", "average_absolute") \
                and cohort > 200 and r.max_any > T1_MS_MAX_PLAUSIBLE:
            why.append(f"max {r.max_any:.0f} > {T1_MS_MAX_PLAUSIBLE} ms")
        reasons.append("; ".join(why))
    subj["flag"] = reasons
    by_group = subj.groupby("group").agg(
        n=("typical", "size"), median=("typical", "median"), min=("typical", "min"),
        max=("typical", "max"), flagged=("flag", lambda s: int((s != "").sum())))
    return subj, by_group, tissue


# --------------------------------------------------------------------------- figures
def _axes_common(fig, ax, title, subtitle, ylabel, footer, log, bands):
    if log:
        ax.set_yscale("log")
    lo, hi = min(bands), max(bands)
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ticks = [b for b in range(4, N_BANDS + 1, 4) if lo + 2 <= b <= hi - 2]
    ax.set_xticks([lo, *ticks, hi] if hi != lo else [lo])
    ax.set_xlabel("Band (distance from the ventricles →)")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.07, 0.945, title, fontsize=14, color=INK)
    fig.text(0.07, 0.915, textwrap.fill(subtitle, 150), fontsize=8.5, color=MUTED, va="top")
    fig.text(0.07, 0.04, footer, fontsize=8, color=MUTED, family="monospace", va="bottom")


def overview_figure(sub, groups, colors, title, ylabel, footer, kw, log, noun_pl):
    """All groups overlaid: median curve and IQR band only, no boxplots."""
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0.08, 0.24, 0.89, 0.62])
    stats = band_stats(sub[sub.value > 0] if log else sub)
    for g in groups:
        s = stats[stats.group == g].sort_values("band")
        if s.empty:
            continue
        ax.fill_between(s.band, s.q1, s.q3, color=colors[g], alpha=0.14, linewidth=0)
        ax.plot(s.band, s["median"], color=colors[g], linewidth=2, marker="o", markersize=2.5,
                solid_capstyle="round", solid_joinstyle="round")
    if not log and not stats.empty:
        lo = stats.q1.min()
        ax.set_ylim(min(0, lo) if lo >= 0 else lo * 1.1, stats.q3.max() * 1.08)
    if len(groups) > 1:
        ymax = ax.get_ylim()[1]
        for b in kw[kw.q < 0.05].band:
            ax.text(b, ymax, "*", ha="center", va="bottom", fontsize=10, color=INK)
    n_per = sub.dropna(subset=["value"]).groupby("group").row_id.nunique()
    ax.legend(handles=[Patch(facecolor=colors[g] + "33", edgecolor=colors[g],
                             label=f"{g} (n = {n_per.get(g, 0)})") for g in groups],
              loc="best", frameon=False, ncol=min(len(groups), 4))
    _axes_common(fig, ax, title,
                 f"All {noun_pl} overlaid: line = median, shading = IQR (Q1-Q3). "
                 f"* = difference between {noun_pl} in that band (Kruskal-Wallis, BH-FDR q < 0.05).",
                 ylabel, footer, log, sorted(sub.band.unique()))
    return fig


def group_figure(sub, group, color, title, ylabel, footer, log, fliers):
    """One group: boxplot per band + median curve and IQR band."""
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0.08, 0.24, 0.89, 0.62])
    d = sub[sub.group == group].dropna(subset=["value"])
    if log:
        d = d[d.value > 0]
    bands = sorted(d.band.unique())
    bp = ax.boxplot(
        [d.loc[d.band == b, "value"].to_numpy() for b in bands], positions=bands, widths=0.6,
        patch_artist=True, showfliers=fliers, manage_ticks=False,
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
        ax.set_ylim((lo / 1.3, hi * 1.3) if log else (min(0, lo) if lo >= 0 else lo * 1.1, hi * 1.06))
    ax.legend(handles=[Patch(facecolor=color + "26", edgecolor=color,
                             label=f"{group} (n = {d.row_id.nunique()})")],
              loc="best", frameon=False)
    _axes_common(fig, ax, title,
                 "Boxplot per band (whiskers 1.5×IQR); line = median, shading = IQR."
                 + ("" if fliers else " Outliers are not drawn (but are included in all statistics)."),
                 ylabel, footer, log, bands)
    return fig


def text_page(title, lines, fontsize=8.5):
    fig = plt.figure(figsize=A4)
    fig.text(0.06, 0.93, title, fontsize=15, color=INK)
    fig.text(0.06, 0.88, "\n".join(lines), fontsize=fontsize, color=INK, va="top",
             family="monospace", linespacing=1.45)
    return fig


def section_page(title, lines):
    fig = plt.figure(figsize=A4)
    fig.text(0.06, 0.55, title, fontsize=22, color=INK)
    fig.text(0.06, 0.49, "\n".join(lines), fontsize=10, color=MUTED, va="top", linespacing=1.6)
    return fig


def fmt_num(v: float) -> str:
    if pd.isna(v):
        return "-"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def table_page(sub, groups, title, ylabel, kw, footer, noun_pl):
    st = band_stats(sub)
    used = sorted(sub.band.unique())
    idx = np.unique(np.linspace(0, len(used) - 1, min(12, len(used))).round().astype(int))
    bands = [used[i] for i in idx]
    rows, cells = [], []
    for g in groups:
        s = st[st.group == g].set_index("band")
        cells.append([f"{fmt_num(s.loc[b, 'median'])}\n[{fmt_num(s.loc[b, 'q1'])}-{fmt_num(s.loc[b, 'q3'])}]"
                      f"\nn={int(s.loc[b, 'n'])}" if b in s.index else "-" for b in bands])
        rows.append(g)
    kwi = kw.set_index("band").q
    cells.append(["-" if pd.isna(kwi.get(b)) else ("<0.001" if kwi[b] < 0.001 else f"{kwi[b]:.3f}")
                  for b in bands])
    rows.append("KW q (FDR)")
    fig = plt.figure(figsize=A4)
    fig.text(0.06, 0.93, title, fontsize=14, color=INK)
    fig.text(0.06, 0.895, textwrap.fill(
        f"Median [Q1-Q3], n = subjects with a value. Units: {ylabel}. Last row: Kruskal-Wallis "
        f"between {noun_pl}, Benjamini-Hochberg FDR across all drawn bands.", 150),
        fontsize=8.5, color=MUTED, va="top")
    ax = fig.add_axes([0.06, 0.2, 0.9, 0.64])
    ax.axis("off")
    tb = ax.table(cellText=cells, rowLabels=rows, colLabels=[f"band {b}" for b in bands],
                  loc="upper left", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.8)
    tb.scale(1, 2.6)
    for (r, _), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if r == 0:
            c.set_text_props(color=MUTED)
    fig.text(0.06, 0.04, footer, fontsize=8, color=MUTED, family="monospace", va="bottom")
    return fig


# --------------------------------------------------------------------------- report
def build_report(df, out: Path, group_col: str, metric: str, source_name: str,
                 group_label: str, ids: pd.Series, groups_order=None, log=False,
                 fliers=False, noun="centre") -> None:
    noun_pl = noun + "s"
    long, found = to_long(df, group_col, metric)
    groups = [g for g in (groups_order or sorted(long.group.unique())) if g in set(long.group)]
    if len(groups) > len(PALETTE):
        raise SystemExit(f"More than {len(PALETTE)} {noun_pl} - merge small groups or set the order list.")
    colors = {g: PALETTE[i] for i, g in enumerate(groups)}
    long = long[long.group.isin(groups)]
    is_volume = metric.startswith("volume")
    rel = add_relative(long) if is_volume and found["whole_band"]["used"] else None
    n_rows = long.groupby("group").row_id.nunique()
    ylabel, unit_note = infer_axis(metric, long.value)

    variants = []  # (title, ylabel, data, footer)
    for tissue, label in TISSUES.items():
        if not found[tissue]["used"]:
            continue
        sub = long[long.tissue == tissue]
        variants.append((label, ylabel, sub, f"Columns: {col_pattern(metric, tissue)}"))
        if rel is not None and tissue != "whole_band":
            r = rel[rel.tissue == tissue].dropna(subset=["value"])
            if not r.empty and r.groupby("band").value.nunique().gt(1).any():
                variants.append((f"{label} - % of band volume", "% of band volume", r,
                                 f"Numerator:   {col_pattern(metric, tissue)}\n"
                                 f"Denominator: {col_pattern(metric, 'whole_band')}"))
    if not variants:
        raise SystemExit("Nothing to plot - all columns are empty or identical in every row.")

    intro = [
        f"Input:          {source_name}",
        f"Created:        {date.today().isoformat()}",
        f"Map:            {MODALITY.upper()}  (columns map_perivent_{metric}_{MODALITY}_band_*)",
        f"Metric:         {metric} - {METRICS.get(metric, metric)}",
        f"Y axis:         {ylabel}",
        *textwrap.wrap(f"                {unit_note}", 130, subsequent_indent=" " * 16),
        f"Grouping:       {group_label}",
        f"Bands:          {BAND_RANGE[0]}-{BAND_RANGE[1]} (of {N_BANDS})",
        (f"Scale:          {'log' if log else 'linear'}"
         f"{'' if fliers else '; boxplot axes clipped to the whisker range'}"),
        "",
        f"Rows (subjects/sessions) per {noun}:",
        *[f"  {g:<22s} {n_rows.get(g, 0)}" for g in groups],
        "",
        "Plotted tissues (only bands whose values differ between subjects):",
    ]
    skipped = []
    for tissue, label in TISSUES.items():
        f = found[tissue]
        if f["used"]:
            intro.append(f"  {label:<18s} {col_pattern(metric, tissue)}   bands {compress(f['used'])}")
        notes = [f"{what} {compress(f[k])}" for what, k in
                 (("empty", "empty"), ("identical in all rows", "constant"), ("missing", "missing"))
                 if f[k]]
        if notes:
            skipped.append(f"  {label:<18s} " + "; ".join(notes))
    if skipped:
        intro += ["", "Skipped bands (no information - not plotted, not tested):", *skipped]
    intro += [
        "",
        "Abbreviations: NAWM = normal-appearing white matter. NASC and NAWMSC are ROI definitions",
        "of the quant_perivent pipeline - see the pipeline documentation for their exact meaning.",
        *(["% of band volume = 100 × <tissue>_in_band / whole_band (same band and subject)."]
          if rel is not None else []),
        "Missing and non-numeric values are dropped per band; decimal commas are converted.",
        f"Statistics: Kruskal-Wallis between {noun_pl} per band ({noun_pl} with n ≥ 3),",
        "Benjamini-Hochberg FDR across the drawn bands of each tissue.",
    ]

    flagged_csv = None
    with PdfPages(out) as pdf:
        pdf.savefig(text_page(f"Periventricular gradient - T1 {metric}, by {noun}", intro))
        plt.close("all")

        if not is_volume:
            subj, by_group, ref_tissue = plausibility(long, metric, ids)
            flagged = subj[subj.flag != ""].sort_values("ratio_to_cohort", ascending=False)
            lines = [
                f"Subject-level value = median over bands of {TISSUES[ref_tissue]} ({metric}).",
                f"Cohort median: {fmt_num(subj.typical.median())}   |   y-axis: {ylabel}",
                "Flag: < 0.5× or > 2× cohort median (typical of a different unit or map type, e.g. 10× =",
                "values stored in 0.1 ms), or any value above 5000 ms (beyond CSF T1 at 3T - fitting",
                "failure, a non-T1 map, or a non-ms unit).",
                "",
                f"{noun.capitalize():<22s} {'n':>5s} {'median':>10s} {'min':>10s} {'max':>10s} {'flagged':>8s}",
                *[f"{g:<22s} {int(r.n):>5d} {fmt_num(r['median']):>10s} {fmt_num(r['min']):>10s} "
                  f"{fmt_num(r['max']):>10s} {int(r.flagged):>8d}"
                  for g, r in by_group.reindex(groups).dropna(how="all").iterrows()],
                "",
                f"Flagged subjects: {len(flagged)}" + (" (first 25 shown; full list in CSV)"
                                                         if len(flagged) > 25 else ""),
                *[f"  {str(r.ID):<28s} {str(r.group):<14s} {fmt_num(r.typical):>9s}   {r.flag}"
                  for _, r in flagged.head(25).iterrows()],
            ]
            pdf.savefig(text_page("Data plausibility check", lines))
            plt.close("all")
            if len(flagged):
                flagged_csv = out.with_name(out.stem + "_flagged_subjects.csv")
                flagged.drop(columns="row_id").to_csv(flagged_csv, index=False)
            print(f"Plausibility: {len(flagged)} flagged subjects.")
            if len(flagged):
                print(by_group.to_string())

        pdf.savefig(section_page(f"1  Overview - all {noun_pl}", [
            f"Median curves and IQR bands of all {noun_pl} in one plot, no boxplots.",
            f"Metric: {metric} - {METRICS.get(metric, metric)}; y axis: {ylabel}",
            f"Each plot is followed by a table of median [IQR] and a test between {noun_pl}.",
            f"{noun_pl.capitalize()}: " + ", ".join(groups)]))
        for title, yl, sub, footer in variants:
            kw = kruskal_by_band(sub)
            pdf.savefig(overview_figure(sub, groups, colors, f"{title} - all {noun_pl}", yl,
                                        footer, kw, log, noun_pl))
            pdf.savefig(table_page(sub, groups, f"{title} - table", yl, kw, footer, noun_pl))
            plt.close("all")

        for i, g in enumerate(groups, start=2):
            pdf.savefig(section_page(f"{i}  {noun.capitalize()} {g}", [
                f"Rows (subjects/sessions): {n_rows.get(g, 0)}",
                "Boxplots per band with median curve and IQR band."]))
            for title, yl, sub, footer in variants:
                if sub[sub.group == g].value.notna().sum() == 0:
                    continue
                pdf.savefig(group_figure(sub, g, colors[g], f"{g}: {title}", yl, footer, log, fliers))
                plt.close("all")
        pdf.infodict()["Title"] = f"Periventricular gradient T1 {metric} by {noun}"
    if flagged_csv:
        print(f"Flagged subjects saved: {flagged_csv}")


def run_report(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    group_col: str | None = None,
    group_from_id: str | None = None,
    group_map: dict[str, str] | None = None,
    na_label: str | None = None,
    drop_groups: list[str] | None = None,
    band_min: int = 1,
    band_max: int = N_BANDS,
    groups: list[str] | None = None,
    group_noun: str = "centre",
    metric: str = "median",
    log: bool = False,
    fliers: bool = False,
) -> Path:
    """Build the PDF report. Callable directly from Python.

    input_path     CSV or XLSX in wide format
    output_path    output PDF; None = next to the input as <name>_perivent_T1_<metric>_by_<noun>.pdf
    group_col      column holding the group (e.g. 'Phenotype', 'centre')
    group_from_id  alternative: column with IDs; group = characters before the first '-'
                   ('PRG-0012' -> 'PRG')
    group_map      merge/rename groups, e.g. {'EarlyMS_dmt': 'EarlyMS'}
    na_label       name for rows with a missing group (e.g. 'pseudo_HC'); None = drop those rows
    drop_groups    categories to exclude, e.g. ['CIS', 'other']; matches the original value
                   or the name after group_map / na_label (so 'EarlyMS_dmt' drops only that
                   subgroup, 'EarlyMS' drops the merged group)
    band_min/max   analyse only bands band_min..band_max (e.g. band_max=30); applies to plots,
                   tables, tests (FDR over these bands only) and the plausibility check
    groups         order of groups = order of colours; groups not listed are dropped; None = alphabetical
    group_noun     word used in the report ('centre', 'phenotype')
    metric         median | average | standard_deviation | median_absolute | average_absolute |
                   volume_absolute_greater_2 | relative_volume_absolute_greater_2
    log            log y axis
    fliers         draw outliers in the boxplots
    """
    global BAND_RANGE
    if not 1 <= band_min <= band_max <= N_BANDS:
        raise SystemExit(f"Need 1 <= band_min <= band_max <= {N_BANDS}, got {band_min}..{band_max}.")
    BAND_RANGE = (band_min, band_max)
    if metric not in METRICS:
        raise SystemExit(f"Unknown metric '{metric}'. Use one of: {', '.join(METRICS)}")
    input_path = Path(input_path)
    output_path = (Path(output_path) if output_path else input_path.with_name(
        f"{input_path.stem}_perivent_T1_{metric}_by_{group_noun}.pdf"))
    df = read_table(input_path).copy()

    if group_from_id:
        if group_from_id not in df.columns:
            raise SystemExit(f"Column '{group_from_id}' not found. First columns: "
                             f"{', '.join(map(str, df.columns[:10]))}")
        ids = df[group_from_id].astype("string").str.strip()
        no_dash = ids.notna() & ~ids.str.contains("-", regex=False)
        if no_dash.any():
            print(f"Note: {int(no_dash.sum())} IDs without '-' (e.g. {ids[no_dash].iloc[0]}) have no group.")
        raw = ids.where(~no_dash).str.split("-", n=1).str[0].str.strip()
        label = f"characters before '-' in column {group_from_id}"
    elif group_col:
        if group_col not in df.columns:
            similar = [c for c in df.columns if group_col.lower()[:5] in str(c).lower()]
            raise SystemExit(f"Column '{group_col}' not found." + (f" Similar: {similar}" if similar else ""))
        raw = df[group_col].astype("string").str.strip()
        label = f"column {group_col}"
    else:
        raise SystemExit("Set group_col or group_from_id.")

    raw = raw.replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    grp = raw.replace(group_map or {})
    if na_label is not None:
        grp = grp.fillna(na_label)
    if drop_groups:
        drop_set = {str(g).strip() for g in drop_groups}
        unknown = drop_set - set(raw.dropna()) - set(grp.dropna())
        if unknown:
            print(f"Note: drop_groups not found in the data: {sorted(unknown)}")
        # matches original names as well as names after merging / NA labelling
        grp = grp.where(~(grp.isin(drop_set) | raw.isin(drop_set)))
    if groups:
        grp = grp.where(grp.isin(groups))
    df["_group"] = grp
    if group_map:
        label += "; merged: " + ", ".join(f"{k} → {v}" for k, v in group_map.items())
    if na_label is not None:
        label += f"; missing → {na_label}"
    if drop_groups:
        label += "; dropped: " + ", ".join(map(str, drop_groups))
    if group_map or na_label is not None or drop_groups:
        print("Group mapping (original → used):")
        print(pd.crosstab(raw.fillna("<NA>"), grp.fillna("<dropped>")).to_string())
    n_drop = int(grp.isna().sum())
    if n_drop:
        print(f"Note: {n_drop} rows without a group are dropped.")
    print(f"{group_noun.capitalize()}s:", grp.value_counts().to_dict())

    if "ID" in df.columns:
        ids_out = df["ID"].astype(str)
    elif {"subject", "session"} <= set(df.columns):
        ids_out = df["subject"].astype(str) + "_" + df["session"].astype(str)
    else:
        ids_out = pd.Series(df.index.astype(str), index=df.index)

    build_report(df, output_path, "_group", metric, input_path.name, label, ids_out,
                 groups, log, fliers, group_noun)
    print(f"Done: {output_path}")
    return output_path


if __name__ == "__main__":
    # ---- settings -----------------------------------------------------------
    INPUT = r"E:\Project_DCE\_DATA_for_merge\combine_results_from_centers_20261116_1340.csv"
    OUTPUT = None               # None = next to the CSV as <name>_perivent_T1_<metric>_by_phenotype.pdf
    PHENOTYPE_COL = "EDSS_category"
    PHENOTYPE_MAP = {           # merge phenotypes before the analysis
        "EarlyMS_no_dmt": "EarlyMS",
        "EarlyMS_dmt": "EarlyMS",
        "Early MS": "EarlyMS",
    }
    NA_LABEL = "pseudo_HC"      # rows with missing phenotype; None = drop them
    DROP = ["NMOSD_MOGAD", "pseudo_HC", "HC"]                   # categories to exclude, e.g. ["CIS", "unknown"]; [] = keep all
    BAND_MIN, BAND_MAX = 1, 30  # analyse only these bands (1-48)
    PHENOTYPES = None           # order/colours, e.g. ["pseudo_HC", "EarlyMS", ...]; None = alphabetical
    METRIC = "median_absolute"       # median | average | standard_deviation | volume_absolute_greater_2 ...
    LOG = False                 # True = log y axis
    FLIERS = False              # True = draw outliers in boxplots
    # ------------------------------------------------------------------------
    run_report(INPUT, OUTPUT, group_col=PHENOTYPE_COL, group_map=PHENOTYPE_MAP,
               na_label=NA_LABEL, drop_groups=DROP, band_min=BAND_MIN, band_max=BAND_MAX, groups=PHENOTYPES, group_noun="EDSS_category",
               metric=METRIC, log=LOG, fliers=FLIERS)