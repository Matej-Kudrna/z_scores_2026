#!/usr/bin/env python3
"""
Výpočet kvantitativní T1 mapy z MP2RAGE dat.

Vychází z analytického modelu MP2RAGE signálu (Marques et al., NeuroImage 2010).
Princip:
  - UNI obraz MP2RAGE = real( INV1* . INV2 / (|INV1|^2 + |INV2|^2) ) -> hodnoty v <-0.5, 0.5>
  - Pro dané sekvenční parametry je vztah UNI(T1) monotónní a jednoznačně invertovatelný.
  - Vytvoříme lookup table UNI -> T1 a interpolujeme.

POZOR: skript NEumí vyčíst sekvenční parametry z .nii. Musíš je doplnit z protokolu
       / JSON sidecaru níže v sekci PARAMETRY. Hodnoty zde jsou jen příklad pro Philips.

Použití:
  python3 mp2rage_t1map.py acq-CS_uni_MP2RAGE.nii -o t1map.nii.gz
  # volitelně INV1/INV2 pro maskování pozadí:
  python3 mp2rage_t1map.py acq-CS_uni_MP2RAGE.nii \
        --inv1 acq-CS_inv-1_MP2RAGE.nii --inv2 acq-CS_inv-2_MP2RAGE.nii -o t1map.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib

# =============================================================================
# PARAMETRY SEKVENCE  -- UPRAV PODLE SVÉHO PROTOKOLU (Philips MR 7700 / Skyra)
# =============================================================================
# Časy v sekundách, úhly ve stupních.
MP2RAGE_TR = 5.0        # MP2RAGE TR [s]  (čas mezi dvěma inverzemi)
TI1        = 0.7        # inversion time 1 [s]
TI2        = 2.5        # inversion time 2 [s]
FA1        = 4.0        # flip angle bloku 1 [deg]
FA2        = 5.0        # flip angle bloku 2 [deg]
TR_FLASH   = 0.0062     # TR jednoho FLASH excitačního pulzu [s]
N_SLICES   = 176        # počet excitací na inverzní blok (== matrix v partition směru / partial Fourier)
SLICE_PE   = 0.5        # pozice "center of k-space" (0.5 = lineární středění); běžně 0.5
INV_EFF    = 0.96       # invertovací účinnost (Philips ~0.96, Siemens ~0.96)
# =============================================================================


def mp2rage_signal(T1, MP2RAGE_TR, TI1, TI2, FA1, FA2, TR_FLASH,
                   n, slice_pe, inv_eff):
    """Vrací hodnotu UNI MP2RAGE signálu pro dané T1 (Marques 2010, eqs. 1-5)."""
    fa1 = np.deg2rad(FA1)
    fa2 = np.deg2rad(FA2)
    E1 = np.exp(-TR_FLASH / T1)

    # počet excitací před / po středu k-prostoru
    n_before = int(round(n * slice_pe))
    n_after = n - n_before

    # časové úseky mezi inverzí a bloky readoutu
    TA = TI1 - n_before * TR_FLASH                 # od inverze k začátku 1. bloku
    TB = TI2 - TI1 - n * TR_FLASH                  # mezi koncem 1. a začátkem 2. bloku
    TC = MP2RAGE_TR - TI2 - n_after * TR_FLASH     # od konce 2. bloku do konce TR

    EA = np.exp(-TA / T1)
    EB = np.exp(-TB / T1)
    EC = np.exp(-TC / T1)

    cosE1_1 = np.cos(fa1) * E1
    cosE1_2 = np.cos(fa2) * E1

    # ustálená podélná magnetizace před inverzí (mzss)
    num = ((((1 - EA) * cosE1_1 ** n + (1 - E1) * (1 - cosE1_1 ** n) / (1 - cosE1_1))
            * EB + (1 - EB)) * cosE1_2 ** n
           + (1 - E1) * (1 - cosE1_2 ** n) / (1 - cosE1_2)) * EC + (1 - EC)
    den = 1 + inv_eff * (cosE1_1 ** n) * (cosE1_2 ** n) * EA * EB * EC
    mzss = num / den

    # signál v 1. bloku (GRE1, prostřední k-space řádek)
    s1 = (((-inv_eff * mzss * EA + (1 - EA)) * cosE1_1 ** n_before
           + (1 - E1) * (1 - cosE1_1 ** n_before) / (1 - cosE1_1)))
    GRE1 = np.sin(fa1) * s1

    # signál v 2. bloku (GRE2)
    s2 = ((mzss - (1 - EC)) / (EC * cosE1_2 ** n_after)
          - (1 - E1) * (cosE1_2 ** (-n_after) - 1) / (1 - cosE1_2))
    GRE2 = np.sin(fa2) * s2

    # UNI obraz
    return (GRE1 * GRE2) / (GRE1 ** 2 + GRE2 ** 2)


def build_lut(p, n_t1=4000, t1_min=0.05, t1_max=6.0):
    """Lookup table T1 -> UNI; ořízne na monotónní rozsah pro jednoznačnou inverzi."""
    T1 = np.linspace(t1_min, t1_max, n_t1)
    uni = np.array([mp2rage_signal(t, p["MP2RAGE_TR"], p["TI1"], p["TI2"],
                                   p["FA1"], p["FA2"], p["TR_FLASH"],
                                   p["N_SLICES"], p["SLICE_PE"], p["INV_EFF"])
                    for t in T1])
    # UNI je monotónně klesající funkce T1 -> seřaď podle uni vzestupně pro interpolaci
    order = np.argsort(uni)
    return uni[order], T1[order]


def _load_json_for(nii_path):
    """Najde JSON sidecar k .nii / .nii.gz a vrátí dict (nebo {} pokud chybí)."""
    import json, os
    base = str(nii_path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    j = base + ".json"
    if os.path.exists(j):
        with open(j) as f:
            return json.load(f)
    return {}


def resolve_params(args):
    """Sestaví parametry sekvence: priorita CLI > UNI sidecar > INV sidecary > default v hlavičce."""
    uni_j = _load_json_for(args.uni)
    inv1_j = _load_json_for(args.inv1) if args.inv1 else {}
    inv2_j = _load_json_for(args.inv2) if args.inv2 else {}

    def pick(*vals, default=None):
        for v in vals:
            if v is not None:
                return v
        return default

    p = {
        "MP2RAGE_TR": pick(args.mp2rage_tr, uni_j.get("RepetitionTime"),
                           inv1_j.get("RepetitionTime"), default=MP2RAGE_TR),
        "TI1": pick(args.ti1, inv1_j.get("InversionTime"), default=TI1),
        "TI2": pick(args.ti2, inv2_j.get("InversionTime"), default=TI2),
        "FA1": pick(args.fa1, inv1_j.get("FlipAngle"), default=FA1),
        "FA2": pick(args.fa2, inv2_j.get("FlipAngle"), default=FA2),
        "TR_FLASH": pick(args.tr_flash,
                         uni_j.get("RepetitionTimeExcitation"),
                         inv1_j.get("RepetitionTimeExcitation"), default=TR_FLASH),
        "N_SLICES": pick(args.n_excitations, default=N_SLICES),
        "SLICE_PE": SLICE_PE,
        "INV_EFF": pick(args.inv_eff, default=INV_EFF),
    }
    print("Parametry sekvence:")
    for k in ("MP2RAGE_TR", "TI1", "TI2", "FA1", "FA2", "TR_FLASH", "N_SLICES", "INV_EFF"):
        print(f"  {k:11s} = {p[k]}")
    return p


def detect_uni_range(data):
    """Vrátí True pokud je UNI v rozsahu cca <-0.5,0.5>, jinak se přeškáluje z int rozsahu."""
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmin >= -0.55 and dmax <= 0.55:
        return data.astype(np.float64)
    # Philips/Siemens často exportují UNI jako uint12/uint16 0..4095 -> přeškáluj na -0.5..0.5
    return (data.astype(np.float64) / dmax) - 0.5


def main():
    ap = argparse.ArgumentParser(description="MP2RAGE -> T1 mapa")
    ap.add_argument("uni", help="UNI MP2RAGE .nii(.gz)")
    ap.add_argument("--inv1", help="INV1 (pro maskování pozadí)", default=None)
    ap.add_argument("--inv2", help="INV2 (pro maskování pozadí)", default=None)
    ap.add_argument("-o", "--out", default="t1map.nii.gz", help="výstupní T1 mapa [ms]")
    ap.add_argument("--unit", choices=["ms", "s"], default="ms",
                    help="jednotky výstupu (default ms)")
    # přepisy parametrů sekvence (jinak se berou z JSON sidecarů / defaultů v hlavičce)
    ap.add_argument("--mp2rage-tr", type=float, default=None, help="[s]")
    ap.add_argument("--ti1", type=float, default=None, help="[s]")
    ap.add_argument("--ti2", type=float, default=None, help="[s]")
    ap.add_argument("--fa1", type=float, default=None, help="[deg]")
    ap.add_argument("--fa2", type=float, default=None, help="[deg]")
    ap.add_argument("--tr-flash", type=float, default=None,
                    help="[s] TR FLASH excitace (často NENÍ v Philips JSON -> zadej z protokolu)")
    ap.add_argument("--n-excitations", type=int, default=None,
                    help="počet excitací na inverzní blok (z protokolu)")
    ap.add_argument("--inv-eff", type=float, default=None, help="invertovací účinnost")
    args = ap.parse_args()

    params = resolve_params(args)

    uni_img = nib.load(args.uni)
    uni = detect_uni_range(np.asarray(uni_img.get_fdata()))

    uni_lut, t1_lut = build_lut(params)

    # interpolace UNI -> T1 (s)
    t1 = np.interp(uni.ravel(), uni_lut, t1_lut,
                   left=t1_lut[0], right=t1_lut[-1]).reshape(uni.shape)

    # maska pozadí pomocí INV2 (lepší SNR než INV1)
    if args.inv2 or args.inv1:
        ref = nib.load(args.inv2 or args.inv1).get_fdata()
        thr = 0.05 * np.nanmax(np.abs(ref))
        t1[np.abs(ref) < thr] = 0.0

    if args.unit == "ms":
        t1 = t1 * 1000.0

    out = nib.Nifti1Image(t1.astype(np.float32), uni_img.affine, uni_img.header)
    out.header.set_data_dtype(np.float32)
    nib.save(out, args.out)
    valid = t1[t1 > 0]
    print(f"Uloženo: {args.out}  [{args.unit}]")
    if valid.size:
        print(f"T1 medián (tkáň): {np.median(valid):.1f} {args.unit}  "
              f"(rozsah {np.percentile(valid,1):.0f}–{np.percentile(valid,99):.0f})")


if __name__ == "__main__":
    main()
