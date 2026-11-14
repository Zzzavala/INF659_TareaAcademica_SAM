"""Diagnostico exploratorio SAM + LoRA contra U-Net. NO entrena nada y NO toca test ni Bolivia.

    python tools/diagnostico_sam_vs_unet.py            # todo (usa la GPU unos minutos para predecir en train)
    python tools/diagnostico_sam_vs_unet.py --sin-gpu  # solo lo que sale de las probabilidades guardadas de valid

Preguntas que responde (todas sobre valid o train, con los modelos ya entrenados):
  D1. Ajuste: IoU de cada modelo en TRAIN (inferencia limpia, sin aumentos) contra valid. Si son parecidos,
      el limite no es el sobreajuste.
  D2. Complementariedad: correlacion del IoU por chip entre modelos, errores compartidos, promedio SAM + U-Net
      y cota "oraculo" de elegir por chip el mejor.
  D3. Donde falla cada uno: por tamano del cuerpo de agua, cerca del borde, y segun si el agua se ve oscura
      en el radar (VH filtrado bajo el umbral fijo de train).
  D4. Cota por resolucion: IoU maximo si la etiqueta se representara a 256 x 256 o 64 x 64.
Salida: results/hacia_060/exploracion/diagnostico_sam_vs_unet.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from inf659sam import config, data, stats  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from robustez_umbral import vh_filtrado  # noqa: E402

HX = config.RESULTS_DIR / "hacia_060"
OUT = HX / "exploracion" / "diagnostico_sam_vs_unet.json"
UMBRAL_DB = -23.3
N_BOOT = 5000
TAM = ((1, 100, "< 100 px"), (100, 1000, "100 a 1 000 px"), (1000, 10000, "1 000 a 10 000 px"), (10000, 10**9, ">= 10 000 px"))
AGUA = ((0, 1, "< 1 %"), (1, 10, "1 a 10 %"), (10, 30, "10 a 30 %"), (30, 101, "> 30 %"))


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fila(pred, gt, lv) -> dict:
    p, g = pred & lv, gt & lv
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    u = tp + fp + fn
    return {"iou": tp / u if u else None, "tp": tp, "fp": fp, "fn": fn, "union": u}


def resumen(rows) -> dict:
    a = stats.aggregate(rows)
    return {k: a[k] for k in ("global_iou", "mean_chip_iou", "precision", "recall")}


def mascaras(probs) -> list[np.ndarray]:
    return [np.zeros((512, 512), bool) if p is None else np.asarray(p, np.float32) > 0.5 for p in probs]


def cargar_valid(chips, semillas, tta=True):
    tag = "_probs_tta_valid.npz" if tta else "_probs_valid.npz"
    sam = [L.load_probs(L.MODEL_DIR / f"lora_db3_otsu_seed{s}{tag}", len(chips)) for s in semillas]
    unet = [L.load_probs(P3.MODEL_DIR / f"unet_db3_seed{s}{tag}", len(chips)) for s in semillas]
    return sam, unet


def prob_o_cero(p):
    return np.zeros((512, 512), np.float32) if p is None else np.asarray(p, np.float32)


def d2_d3_d4(chips) -> dict:
    out: dict = {}
    etiquetas = []
    for c in chips:
        d = data.load_chip(c)
        lab = d["label"]
        vh, sv = vh_filtrado(d, True)
        oscuro = (vh < UMBRAL_DB) & sv if vh is not None else np.zeros_like(sv)
        etiquetas.append({"gt": lab == 1, "lv": lab != -1, "oscuro": oscuro, "region": data.region_of(c)})
    regiones = [e["region"] for e in etiquetas]

    sam3, unet3 = cargar_valid(chips, (0, 1, 2))
    sam5, unet5 = cargar_valid(chips, (0, 1, 2, 3, 4))
    es, eu = L.ensemble(sam3), L.ensemble(unet3)
    es5, eu5 = L.ensemble(sam5), L.ensemble(unet5)
    ms, mu = mascaras(es), mascaras(eu)
    comb3 = [(prob_o_cero(a) + prob_o_cero(b)) / 2 > 0.5 for a, b in zip(es, eu)]
    comb5 = [(prob_o_cero(a) + prob_o_cero(b)) / 2 > 0.5 for a, b in zip(es5, eu5)]

    def filas(masks):
        return [fila(m, e["gt"], e["lv"]) for m, e in zip(masks, etiquetas)]

    rs, ru, rc3, rc5 = filas(ms), filas(mu), filas(comb3), filas(comb5)
    rs5, ru5 = filas(mascaras(es5)), filas(mascaras(eu5))
    out["valid_ens3_tta"] = {"sam": resumen(rs), "unet": resumen(ru), "sam_mas_unet": resumen(rc3)}
    out["valid_ens5_tta"] = {"sam": resumen(rs5), "unet": resumen(ru5), "sam_mas_unet": resumen(rc5)}

    def comparar(a, b):
        return {"por_chip": stats.paired_bootstrap(a, b, n_boot=N_BOOT)["global_iou"],
                "por_region": stats.paired_bootstrap(a, b, groups=regiones, n_boot=N_BOOT)["global_iou"],
                "wilcoxon": stats.wilcoxon_paired([r["iou"] for r in a], [r["iou"] for r in b])}

    out["combinado_vs_unet_ens3"] = comparar(ru, rc3)
    out["combinado_vs_sam_ens3"] = comparar(rs, rc3)
    out["combinado_vs_unet_ens5"] = comparar(ru5, rc5)

    # Correlacion por chip y oraculo
    ok = [i for i in range(len(chips)) if rs[i]["iou"] is not None and ru[i]["iou"] is not None]
    a, b = np.array([rs[i]["iou"] for i in ok]), np.array([ru[i]["iou"] for i in ok])
    out["correlacion_iou_por_chip"] = {"n": len(ok), "pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0]),
                                       "dif_abs_mediana": float(np.median(np.abs(a - b))),
                                       "chips_con_dif_mayor_0.10": int((np.abs(a - b) > 0.10).sum())}
    mejor = [x if (x["iou"] or 0) >= (y["iou"] or 0) else y for x, y in zip(rs, ru)]
    out["oraculo_mejor_por_chip"] = resumen(mejor)

    # Errores compartidos (pixeles)
    cnt = dict.fromkeys(("fn_ambos", "fn_solo_sam", "fn_solo_unet", "fp_ambos", "fp_solo_sam", "fp_solo_unet"), 0)
    tam = {n: {"agua": 0, "tp_sam": 0, "tp_unet": 0, "comp": 0, "comp_sam": 0, "comp_unet": 0} for *_, n in TAM}
    borde = {k: {"err_borde": 0, "err_total": 0} for k in ("sam", "unet")}
    px_borde = px_total = 0
    obs = {k: {"agua": 0, "tp_sam": 0, "tp_unet": 0} for k in ("agua_oscura", "agua_clara")}
    obs |= {k: {"tierra": 0, "fp_sam": 0, "fp_unet": 0} for k in ("tierra_oscura", "tierra_clara")}
    res = {256: [], 64: []}
    por_region: dict = {}
    por_agua = {n: {"sam": [], "unet": []} for *_, n in AGUA}
    for i, e in enumerate(etiquetas):
        gt, lv = e["gt"] & e["lv"], e["lv"]
        s, u = ms[i] & lv, mu[i] & lv
        fn_s, fn_u, fp_s, fp_u = gt & ~s, gt & ~u, s & ~gt, u & ~gt
        cnt["fn_ambos"] += int((fn_s & fn_u).sum())
        cnt["fn_solo_sam"] += int((fn_s & ~fn_u).sum())
        cnt["fn_solo_unet"] += int((fn_u & ~fn_s).sum())
        cnt["fp_ambos"] += int((fp_s & fp_u).sum())
        cnt["fp_solo_sam"] += int((fp_s & ~fp_u).sum())
        cnt["fp_solo_unet"] += int((fp_u & ~fp_s).sum())
        # Tamano del cuerpo de agua (componentes 8-conexos de la etiqueta)
        comp, n = ndi.label(gt, structure=np.ones((3, 3)))
        if n:
            sizes = np.bincount(comp.ravel())[1:]
            tp_s = np.bincount(comp[s & gt].ravel(), minlength=n + 1)[1:]
            tp_u = np.bincount(comp[u & gt].ravel(), minlength=n + 1)[1:]
            for lo, hi, nom in TAM:
                m = (sizes >= lo) & (sizes < hi)
                t = tam[nom]
                t["agua"] += int(sizes[m].sum())
                t["tp_sam"] += int(tp_s[m].sum())
                t["tp_unet"] += int(tp_u[m].sum())
                t["comp"] += int(m.sum())
                t["comp_sam"] += int((tp_s[m] >= 0.5 * sizes[m]).sum())
                t["comp_unet"] += int((tp_u[m] >= 0.5 * sizes[m]).sum())
        # Franja de 2 px a cada lado del borde agua/no agua
        if gt.any() and (~gt).any():
            franja = ((ndi.distance_transform_edt(gt) <= 2) & gt) | ((ndi.distance_transform_edt(~gt) <= 2) & ~gt)
        else:
            franja = np.zeros_like(gt)
        franja &= lv
        px_borde += int(franja.sum())
        px_total += int(lv.sum())
        for k, err in (("sam", fn_s | fp_s), ("unet", fn_u | fp_u)):
            borde[k]["err_borde"] += int((err & franja).sum())
            borde[k]["err_total"] += int(err.sum())
        # ¿El agua se ve oscura en el radar?
        osc = e["oscuro"]
        for nom, m in (("agua_oscura", gt & osc), ("agua_clara", gt & ~osc)):
            obs[nom]["agua"] += int(m.sum())
            obs[nom]["tp_sam"] += int((m & s).sum())
            obs[nom]["tp_unet"] += int((m & u).sum())
        tierra = lv & ~gt
        for nom, m in (("tierra_oscura", tierra & osc), ("tierra_clara", tierra & ~osc)):
            obs[nom]["tierra"] += int(m.sum())
            obs[nom]["fp_sam"] += int((m & s).sum())
            obs[nom]["fp_unet"] += int((m & u).sum())
        # Cota por resolucion
        g = torch.from_numpy(gt.astype(np.float32))[None, None]
        for r in res:
            abajo = F.interpolate(g, size=(r, r), mode="bilinear", antialias=True, align_corners=False)
            arriba = F.interpolate(abajo, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy() > 0.5
            res[r].append(fila(arriba, gt, lv))
        por_region.setdefault(e["region"], {"sam": [], "unet": []})
        por_region[e["region"]]["sam"].append(rs[i])
        por_region[e["region"]]["unet"].append(ru[i])
        pct = 100 * gt.sum() / max(lv.sum(), 1)
        for lo, hi, nom in AGUA:
            if lo <= pct < hi:
                por_agua[nom]["sam"].append(rs[i])
                por_agua[nom]["unet"].append(ru[i])

    fn_tot_s, fn_tot_u = cnt["fn_ambos"] + cnt["fn_solo_sam"], cnt["fn_ambos"] + cnt["fn_solo_unet"]
    fp_tot_s, fp_tot_u = cnt["fp_ambos"] + cnt["fp_solo_sam"], cnt["fp_ambos"] + cnt["fp_solo_unet"]
    out["errores_compartidos"] = cnt | {
        "pct_fn_de_unet_que_tambien_falla_sam": 100 * cnt["fn_ambos"] / max(fn_tot_u, 1),
        "pct_fn_de_sam_que_tambien_falla_unet": 100 * cnt["fn_ambos"] / max(fn_tot_s, 1),
        "pct_fp_de_unet_que_tambien_marca_sam": 100 * cnt["fp_ambos"] / max(fp_tot_u, 1),
        "pct_fp_de_sam_que_tambien_marca_unet": 100 * cnt["fp_ambos"] / max(fp_tot_s, 1)}
    agua_total = sum(t["agua"] for t in tam.values())
    out["por_tamano_del_cuerpo_de_agua"] = {
        n: {"pct_del_agua": 100 * t["agua"] / max(agua_total, 1), "componentes": t["comp"],
            "recall_sam": t["tp_sam"] / max(t["agua"], 1), "recall_unet": t["tp_unet"] / max(t["agua"], 1),
            "pct_componentes_detectados_sam": 100 * t["comp_sam"] / max(t["comp"], 1),
            "pct_componentes_detectados_unet": 100 * t["comp_unet"] / max(t["comp"], 1)} for n, t in tam.items()}
    out["borde_2px"] = {"pct_de_pixeles_en_la_franja": 100 * px_borde / max(px_total, 1)} | {
        k: {"pct_de_sus_errores_en_la_franja": 100 * v["err_borde"] / max(v["err_total"], 1), "errores": v["err_total"]}
        for k, v in borde.items()}
    agua_px = obs["agua_oscura"]["agua"] + obs["agua_clara"]["agua"]
    tierra_px = obs["tierra_oscura"]["tierra"] + obs["tierra_clara"]["tierra"]
    out["visibilidad_en_radar"] = {"umbral_db": UMBRAL_DB}
    for nom in ("agua_oscura", "agua_clara"):
        o = obs[nom]
        out["visibilidad_en_radar"][nom] = {"pct_del_agua": 100 * o["agua"] / max(agua_px, 1),
                                            "recall_sam": o["tp_sam"] / max(o["agua"], 1),
                                            "recall_unet": o["tp_unet"] / max(o["agua"], 1)}
    for nom in ("tierra_oscura", "tierra_clara"):
        o = obs[nom]
        out["visibilidad_en_radar"][nom] = {"pct_de_la_tierra": 100 * o["tierra"] / max(tierra_px, 1),
                                            "tasa_fp_sam": o["fp_sam"] / max(o["tierra"], 1),
                                            "tasa_fp_unet": o["fp_unet"] / max(o["tierra"], 1)}
    # Parte de los errores de cada modelo que cae donde el radar contradice a la etiqueta:
    # agua omitida que no se ve oscura, o falsa alarma sobre tierra que si se ve oscura.
    for k, fn_tot, fp_tot in (("sam", fn_tot_s, fp_tot_s), ("unet", fn_tot_u, fp_tot_u)):
        fn_clara = obs["agua_clara"]["agua"] - obs["agua_clara"][f"tp_{k}"]
        fp_oscura = obs["tierra_oscura"][f"fp_{k}"]
        out["visibilidad_en_radar"][f"errores_{k}"] = {
            "pct_fn_en_agua_clara": 100 * fn_clara / max(fn_tot, 1),
            "pct_fp_en_tierra_oscura": 100 * fp_oscura / max(fp_tot, 1),
            "pct_de_todos_sus_errores_donde_el_radar_contradice": 100 * (fn_clara + fp_oscura) / max(fn_tot + fp_tot, 1)}
    # Cota simple: detectar toda el agua oscura, nada de la clara y ninguna falsa alarma.
    out["visibilidad_en_radar"]["iou_si_solo_se_detectara_el_agua_oscura"] = obs["agua_oscura"]["agua"] / max(agua_px, 1)
    out["cota_por_resolucion"] = {str(r): resumen(v)["global_iou"] for r, v in res.items()}
    out["por_region"] = {r: {"chips": len(v["sam"]), "sam": stats.aggregate(v["sam"])["global_iou"],
                             "unet": stats.aggregate(v["unet"])["global_iou"]} for r, v in sorted(por_region.items())}
    out["por_pct_de_agua"] = {n: {"chips": len(v["sam"]), "sam_global": stats.aggregate(v["sam"])["global_iou"] if v["sam"] else None,
                                  "unet_global": stats.aggregate(v["unet"])["global_iou"] if v["unet"] else None,
                                  "sam_medio": stats.aggregate(v["sam"])["mean_chip_iou"] if v["sam"] else None,
                                  "unet_medio": stats.aggregate(v["unet"])["mean_chip_iou"] if v["unet"] else None}
                              for n, v in por_agua.items()}
    return out


def d1_ajuste(train, valid) -> dict:
    """IoU en train con inferencia limpia (sin aumentos, sin TTA), por semilla y en ensamble."""
    import evaluar_test as E

    dev = config.get_device()
    items = L.prepare_chips(train, "db3", dev)
    gts = [(it["gt"].cpu().numpy(), it["valid"].cpu().numpy()) for it in items]
    sam_v, unet_v = cargar_valid(valid, (0, 1, 2), tta=False)
    lab_v = []
    for c in valid:
        lab = data.load_chip(c)["label"]
        lab_v.append((lab == 1, lab != -1))
    out = {}
    for key, probs_valid in (("sam", sam_v), ("unet", unet_v)):
        ps, por_semilla = [], []
        for i, sd in enumerate(config.SEEDS):
            torch.cuda.empty_cache()
            model = E.load_model(key, sd, dev)
            p = E.predict(key, model, items, tta=False)
            del model
            ps.append(p)
            tr = resumen([fila(m, g, v) for m, (g, v) in zip(mascaras(p), gts)])
            va = resumen([fila(m, g, v) for m, (g, v) in zip(mascaras(probs_valid[i]), lab_v)])
            por_semilla.append({"semilla": sd, "train": tr, "valid": va})
        ens_tr = resumen([fila(m, g, v) for m, (g, v) in zip(mascaras(L.ensemble(ps)), gts)])
        ens_va = resumen([fila(m, g, v) for m, (g, v) in zip(mascaras(L.ensemble(probs_valid)), lab_v)])
        out[key] = {"por_semilla": por_semilla, "ensamble_3": {"train": ens_tr, "valid": ens_va},
                    "train_media": float(np.mean([x["train"]["global_iou"] for x in por_semilla])),
                    "valid_media": float(np.mean([x["valid"]["global_iou"] for x in por_semilla]))}
    del items
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-gpu", action="store_true")
    a = ap.parse_args()
    valid = j(HX / "paso2" / "config_lora_db3_otsu.json")["chips_valid"]
    train = j(config.RESULTS_DIR / "particiones_usadas.json")["train"]
    out = {"nota": "Exploratorio. Solo train y valid, con modelos ya entrenados; no usa test ni Bolivia.",
           "n_valid": len(valid), "n_train": len(train)}
    out |= d2_d3_d4(valid)
    # Comprobacion: los ensambles reproducen los valores publicados (0.6145 y 0.6422).
    assert abs(out["valid_ens3_tta"]["sam"]["global_iou"] - 0.6145) < 0.003, out["valid_ens3_tta"]["sam"]
    assert abs(out["valid_ens3_tta"]["unet"]["global_iou"] - 0.6422) < 0.003, out["valid_ens3_tta"]["unet"]
    if not a.sin_gpu:
        out["ajuste_train_vs_valid"] = d1_ajuste(train, valid)
    elif OUT.exists() and "ajuste_train_vs_valid" in j(OUT):
        out["ajuste_train_vs_valid"] = j(OUT)["ajuste_train_vs_valid"]  # se conserva la medicion previa con GPU
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    figura(out)
    print(json.dumps({k: out[k] for k in ("visibilidad_en_radar", "cota_por_resolucion")}, indent=1, ensure_ascii=False))


def figura(o: dict):
    """Tres paneles para presentar el diagnostico (results/figuras/diagnostico_sam_vs_unet.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SAM, UNET, INK, MUTED, GRID = "#5B35A8", "#00918D", "#28282D", "#5A5A64", "#E6E1F0"
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": "#B9B4C6", "xtick.color": INK, "ytick.color": MUTED})
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw={"width_ratios": [1, 1.05, 1.45]})

    def barras(a, grupos, vs, vu, ylabel, titulo, ymax=1.0):
        x = np.arange(len(grupos))
        for dx, v, col, nom in ((-0.19, vs, SAM, "SAM + LoRA"), (0.19, vu, UNET, "U-Net")):
            b = a.bar(x + dx, v, 0.36, color=col, label=nom, zorder=2)
            for r in b:
                a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015 * ymax, f"{r.get_height():.2f}", ha="center", fontsize=9)
        a.set_xticks(x, grupos, fontsize=9.5)
        a.set_ylim(0, ymax)
        a.set_ylabel(ylabel)
        a.set_title(titulo, fontsize=10.5, fontweight="bold", loc="left")
        a.yaxis.grid(True, color=GRID, zorder=0)
        a.set_axisbelow(True)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(length=0)

    if "ajuste_train_vs_valid" in o:
        aj = o["ajuste_train_vs_valid"]
        barras(ax[0], ["Train (250 chips)", "Valid (86 chips)"], [aj["sam"]["train_media"], aj["sam"]["valid_media"]],
               [aj["unet"]["train_media"], aj["unet"]["valid_media"]], "IoU global (media de 3 semillas)",
               "A. Ninguno memoriza train", ymax=0.8)
    v = o["visibilidad_en_radar"]
    barras(ax[1], [f"Agua oscura en radar\n({v['agua_oscura']['pct_del_agua']:.0f} % del agua)",
                   f"Agua que no se ve oscura\n({v['agua_clara']['pct_del_agua']:.0f} % del agua)"],
           [v["agua_oscura"]["recall_sam"], v["agua_clara"]["recall_sam"]],
           [v["agua_oscura"]["recall_unet"], v["agua_clara"]["recall_unet"]], "Sensibilidad (agua encontrada)",
           "B. El error está donde el radar no ve el agua", ymax=1.12)
    t = o["por_tamano_del_cuerpo_de_agua"]
    barras(ax[2], [f"{k}\n({d['pct_del_agua']:.0f} % del agua)" for k, d in t.items()], [d["recall_sam"] for d in t.values()],
           [d["recall_unet"] for d in t.values()], "Sensibilidad", "C. SAM pierde más en cuerpos de agua pequeños", ymax=1.0)
    ax[2].set_xlabel("Tamaño del cuerpo de agua en la etiqueta (1 px = 100 m²)", fontsize=9.5)
    ax[0].legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "diagnostico_sam_vs_unet.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
