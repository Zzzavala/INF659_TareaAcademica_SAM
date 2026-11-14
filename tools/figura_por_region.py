"""Figura de la validación cruzada por región (sección 2e de la bitácora).

Lee results/hacia_060/por_region/resumen.json y no entrena ni predice nada.
Salida: results/figuras/validacion_por_region.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "hacia_060" / "por_region" / "resumen.json"
OUT = REPO / "results" / "figuras" / "validacion_por_region.png"

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8984"
TINTA, TINTA2, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    r = json.loads(RES.read_text(encoding="utf-8"))
    pr = r["por_region"]
    regiones = sorted(r["regiones"], key=lambda g: pr[g]["unet"])
    sam = [pr[g]["sam"] for g in regiones]
    unet = [pr[g]["unet"] for g in regiones]
    umb = [pr[g]["umbral"] for g in regiones]
    n = [pr[g]["chips"] for g in regiones]
    diff = [s - u for s, u in zip(sam, unet)]
    comp = r["comparaciones"]["sam_menos_unet"]["por_region"]["global_iou"]
    ag = r["agregado"]

    plt.rcParams.update({"font.size": 10, "text.color": TINTA, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.6, 1]},
                                 facecolor=FONDO)
    y = range(len(regiones))
    h = 0.36
    a1.barh([i + h / 2 for i in y], sam, height=h - 0.04, color=AZUL, label="SAM + LoRA")
    a1.barh([i - h / 2 for i in y], unet, height=h - 0.04, color=NARANJA, label="U-Net")
    a1.scatter(umb, list(y), marker="|", s=380, color=GRIS, linewidths=2, zorder=3,
               label="Umbral fijo VH (referencia)")
    a1.set_yticks(list(y), [f"{g}  (n={k})" for g, k in zip(regiones, n)])
    a1.set_xlim(0, 1)
    a1.set_xlabel("IoU global de la región cuando el modelo no la vio")
    a1.set_title("IoU por región no vista", loc="left", fontweight="bold")
    a1.legend(loc="lower right", frameon=False, fontsize=9)

    colores = [AZUL if d > 0 else NARANJA for d in diff]
    a2.barh(list(y), diff, height=0.55, color=colores, zorder=2)
    a2.axvline(0, color=TINTA2, lw=1)
    lo, hi = comp["ic95"]
    a2.axvspan(lo, hi, color=GRIS, alpha=0.18, lw=0, zorder=0)
    a2.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    a2.axvline(comp["diff_b_menos_a"], color=TINTA, lw=1.5, ls="--")
    a2.set_yticks(list(y), [""] * len(regiones))
    a2.set_xlim(-0.2, 0.2)
    a2.set_xlabel("SAM − U-Net (IoU)")
    a2.set_title("Diferencia por región", loc="left", fontweight="bold")
    a2.text(0.02, -0.16, f"Total 336 chips: SAM {ag['sam']['global_iou']:.3f} · U-Net "
            f"{ag['unet']['global_iou']:.3f}\nDiferencia {comp['diff_b_menos_a']:+.3f}, IC 95 % por región "
            f"[{lo:+.3f}, {hi:+.3f}] (franja gris)",
            transform=a2.transAxes, fontsize=9, color=TINTA2, va="top")
    a2.text(0.98, 0.02, "SAM mejor →", transform=a2.transAxes, ha="right", fontsize=8, color=TINTA2)
    a2.text(0.02, 0.02, "← U-Net mejor", transform=a2.transAxes, ha="left", fontsize=8, color=TINTA2)

    for ax in (a1, a2):
        ax.set_facecolor(FONDO)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Validación cruzada dejando una región fuera (train + valid, 10 regiones, semilla 0)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=FONDO)
    print(OUT)


if __name__ == "__main__":
    main()
