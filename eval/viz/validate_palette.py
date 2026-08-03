#!/usr/bin/env python
"""Python port of the dataviz skill's palette validator (node 12 here can't run
the .js). Computes the color-only checks — never eyeball CVD separation.

Checks (categorical mode): lightness band, chroma floor, CVD separation
(Machado-Oliveira-Fernandes 2009 severity-1.0, protan/deutan, tritan reported),
normal-vision floor, contrast vs surface. Faithful to scripts/validate_palette.js.

Usage:
    python validate_palette.py "#2a78d6,#eb6834" --mode light --pairs all
"""
import argparse, math, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],[0.114503, 0.786281, 0.099216],[-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],[0.280085, 0.672501, 0.047413],[-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],[-0.078411, 0.930809, 0.147602],[0.004733, 0.691367, 0.303900]],
}

def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
def s2lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
def lin(h): return [s2lin(c) for c in hex2srgb(h)]
def relLum(h):
    r, g, b = lin(h); return 0.2126*r + 0.7152*g + 0.0722*b
def contrast(a, b):
    hi, lo = sorted([relLum(a), relLum(b)], reverse=True)
    return (hi+0.05)/(lo+0.05)
def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]
def oklch(h):
    L, a, b = oklab_from_lin(lin(h)); return L, math.hypot(a, b)
def simulate(h, kind):
    r, g, b = lin(h); M = MACHADO[kind]
    return [max(0, min(1, M[i][0]*r + M[i][1]*g + M[i][2]*b)) for i in range(3)]
def deltaE(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100*math.dist(a, b)

def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]; report = []; ok = True
    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not offband
    report.append(("Lightness band", not offband, f"outside: {offband}" if offband else f"all {len(palette)} in L {lo}-{hi}"))
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", not lowc, f"gray: {lowc}" if lowc else f"all >= {CHROMA_FLOOR}"))
    n = len(palette)
    if pairs == "all":
        pl = [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        pl = [(i, i+1) for i in range(n-1)]
    label = "all-pairs" if pairs == "all" else "adjacent"
    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pl:
            d = deltaE(palette[i], palette[j], kind)
            if worst is None or d < worst[0]: worst = (d, kind, palette[i], palette[j])
    wd = worst[0] if worst else 99
    state = "pass" if wd >= CVD_TARGET else ("floor" if wd >= CVD_FLOOR else "fail")
    if state == "fail": ok = False
    report.append(("CVD separation", state, f"worst {label} {worst[3]}<->{worst[2]} dE {wd:.1f} ({worst[1]})" if worst else "n/a"))
    nworst = None
    for i, j in pl:
        d = deltaE(palette[i], palette[j])
        if nworst is None or d < nworst[0]: nworst = (d, palette[i], palette[j])
    nd = nworst[0] if nworst else 99
    nstate = nd >= NORMAL_FLOOR
    ok &= nstate
    report.append(("Normal-vision floor", nstate, f"worst {label} dE {nd:.1f}" + ("" if nstate else f" — below {NORMAL_FLOOR}!")))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "relief" if low else "pass", f"below {CONTRAST_MIN}:1: {low}" if low else f"all >= {CONTRAST_MIN}:1"))
    return report, ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light")
    ap.add_argument("--surface", default=None)
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args()
    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    report, ok = validate(pal, a.mode, a.surface, a.pairs)
    for name, state, msg in report:
        print(f"  [{str(state):5s}] {name:22s} {msg}")
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
