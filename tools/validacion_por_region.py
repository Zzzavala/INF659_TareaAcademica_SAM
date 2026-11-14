"""R4 (bitacora, seccion 2e): validacion cruzada dejando una region fuera. SAM + LoRA contra U-Net.

    python tools/validacion_por_region.py            # 10 pliegues (~5 h en la RTX 3070); se puede cortar y relanzar
    python tools/validacion_por_region.py --rapido   # prueba de humo de pocos minutos (no toca resultados)

Usa solo train + valid (336 chips de 10 regiones). Test y Bolivia NO se usan.
En cada pliegue se entrena con 9 regiones y se predice la que quedo fuera, con la configuracion ya fijada
(`lora_db3_otsu` del paso 2 y `unet_db3` del paso 3a, semilla 0, ultima epoca, umbral 0.5).
Avance: results/hacia_060/por_region/estado.json y ejecucion.log. Al terminar: resumen.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    try:  # mientras este proceso viva, Windows no entra en suspension
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from inf659sam import config, data, stats  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from inf659sam.baseline import otsu_masks  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from inf659sam.hacia060 import _row  # noqa: E402
from robustez_umbral import EDGES, GRID, vh_filtrado  # noqa: E402

OUT = config.RESULTS_DIR / "hacia_060" / "por_region"
CACHE = config.CACHE_DIR / "modelos" / "hacia_060_por_region"
SEED = 0
N_BOOT = 5000
METODOS = ("sam", "sam_tta", "unet", "unet_tta", "otsu", "umbral")


class Registro:
    def __init__(self, out: Path):
        self.out = out
        self.log = open(out / "ejecucion.log", "a", encoding="utf-8")
        self.estado = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "pliegues_terminados": [], "etapa": ""}

    def info(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def set(self, **kw):
        self.estado.update(kw, actualizado=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.out / "estado.json").write_text(json.dumps(self.estado, indent=1, ensure_ascii=False), encoding="utf-8")


def referencias_sin_entrenar(chips):
    """Filas de Otsu por chip, histogramas por chip para elegir el umbral fijo por pliegue y VH filtrado."""
    otsu, hist, vhf = [], [], []
    for c in chips:
        d = data.load_chip(c)
        lab = d["label"]
        gt, lv = lab == 1, lab != -1
        om = otsu_masks(d["vv"], d["vh"])
        p = om["opened"] & lv
        tp = int((p & gt).sum())
        otsu.append(_row(tp, int((p & ~gt).sum()), int((gt & lv).sum()) - tp))
        vh, sv = vh_filtrado(d, True)
        if vh is None:
            hist.append((np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1)))
            vhf.append(None)
            continue
        m = sv & lv
        hist.append((np.histogram(vh[m & gt], EDGES)[0], np.histogram(vh[m & ~gt], EDGES)[0]))
        vhf.append((vh.astype(np.float16), sv))
    return otsu, hist, vhf


def umbral_de(idx_train, hist):
    ha = sum(hist[i][0] for i in idx_train)
    ht = sum(hist[i][1] for i in idx_train)
    tp, fp = np.cumsum(ha)[:-1], np.cumsum(ht)[:-1]
    iou = tp / np.maximum(tp + (ha.sum() - tp) + fp, 1)
    i = int(np.argmax(iou))
    return float(GRID[i])


def filas_umbral(idx, thr, vhf, chips):
    rows = []
    for i in idx:
        lab = data.load_chip(chips[i])["label"]
        gt, lv = lab == 1, lab != -1
        pred = np.zeros_like(gt) if vhf[i] is None else (vhf[i][0].astype(np.float32) < thr) & vhf[i][1] & lv
        tp = int((pred & gt).sum())
        rows.append(_row(tp, int((pred & ~gt).sum()), int((gt & lv).sum()) - tp))
    return rows


def correr_pliegue(region, chips, regiones, items, refs, P, reg: Registro, dev) -> dict:
    idx_te = [i for i, r in enumerate(regiones) if r == region]
    idx_tr = [i for i, r in enumerate(regiones) if r != region]
    tr, te = [items[i] for i in idx_tr], [items[i] for i in idx_te]
    out = {"region": region, "chips": [chips[i] for i in idx_te], "n_train": len(idx_tr), "semilla": SEED,
           "filas": {}, "entrenamiento": {}}

    # SAM + LoRA (paso 2, lora_db3_otsu)
    reg.set(etapa=f"{region}: entrenando SAM + LoRA ({P['epochs_lora']} épocas, {len(tr)} chips)")
    torch.cuda.empty_cache()
    model = L.build_model(dev)

    def ep_log(nombre, total):
        def f(ep, h):
            if nombre == "SAM" or ep % 10 == 0 or ep == total:
                reg.info(f"  [{region} {nombre}] época {ep}/{total} pérdida {h['perdida_media']:.4f} "
                         f"IoU entrenamiento {h['iou_entrenamiento']:.4f} ({h['seg_epoca']:.0f} s)")
            reg.set(detalle=f"{nombre} época {ep}/{total}")
        return f

    h = L.train(model, tr, seed=SEED, prompts="otsu", epochs=P["epochs_lora"], on_epoch=ep_log("SAM", P["epochs_lora"]))
    p_ind = L.predict_probs(model, te, "otsu", tta=False)
    p_tta = L.predict_probs(model, te, "otsu", tta=True)
    L.save_probs(CACHE / f"{region}_sam_probs.npz", p_ind)
    L.save_probs(CACHE / f"{region}_sam_probs_tta.npz", p_tta)
    out["filas"]["sam"] = L.rows_from_probs(p_ind, te, L.THRESHOLD)
    out["filas"]["sam_tta"] = L.rows_from_probs(p_tta, te, L.THRESHOLD)
    out["entrenamiento"]["sam"] = {k: v for k, v in h.items() if k != "historial"} | {
        "iou_entrenamiento_final": h["historial"][-1]["iou_entrenamiento"]}
    del model
    torch.cuda.empty_cache()

    # U-Net desde cero (paso 3a, unet_db3)
    reg.set(etapa=f"{region}: entrenando U-Net ({P['epochs_unet']} épocas, {len(tr)} chips)")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = P3.UNet().to(dev)
    h = P3.train_unet(model, tr, seed=SEED, epochs=P["epochs_unet"], on_epoch=ep_log("U-Net", P["epochs_unet"]))
    p_ind = P3.predict_unet(model, te, tta=False)
    p_tta = P3.predict_unet(model, te, tta=True)
    L.save_probs(CACHE / f"{region}_unet_probs.npz", p_ind)
    L.save_probs(CACHE / f"{region}_unet_probs_tta.npz", p_tta)
    out["filas"]["unet"] = L.rows_from_probs(p_ind, te, P3.THRESHOLD)
    out["filas"]["unet_tta"] = L.rows_from_probs(p_tta, te, P3.THRESHOLD)
    out["entrenamiento"]["unet"] = {k: v for k, v in h.items() if k != "historial"} | {
        "iou_entrenamiento_final": h["historial"][-1]["iou_entrenamiento"]}
    del model
    torch.cuda.empty_cache()

    # Referencias sin entrenamiento
    otsu, hist, vhf = refs
    thr = umbral_de(idx_tr, hist)
    out["umbral_db"] = thr
    out["filas"]["otsu"] = [otsu[i] for i in idx_te]
    out["filas"]["umbral"] = filas_umbral(idx_te, thr, vhf, chips)
    out["iou_region"] = {m: stats.aggregate(out["filas"][m])["global_iou"] for m in METODOS}
    return out


def resumen(chips, regiones, pliegues: dict) -> dict:
    orden = sorted(set(regiones))
    filas = {m: [] for m in METODOS}
    grupos, nombres = [], []
    for r in orden:
        f = pliegues[r]
        for m in METODOS:
            filas[m] += f["filas"][m]
        grupos += [r] * len(f["chips"])
        nombres += f["chips"]
    res = {"nota": "Exploratorio (seccion 2e). Solo train + valid; test y Bolivia no se usan.",
           "n_chips": len(nombres), "regiones": orden, "semilla": SEED,
           "agregado": {m: stats.aggregate(filas[m]) for m in METODOS},
           "por_region": {r: pliegues[r]["iou_region"] | {"chips": len(pliegues[r]["chips"]),
                                                           "umbral_db": pliegues[r]["umbral_db"]} for r in orden}}

    def comparar(a, b):
        ra = [res["por_region"][r][a] for r in orden]
        rb = [res["por_region"][r][b] for r in orden]
        return {"por_region": stats.paired_bootstrap(filas[a], filas[b], groups=grupos, n_boot=N_BOOT),
                "por_chip": stats.paired_bootstrap(filas[a], filas[b], n_boot=N_BOOT),
                "wilcoxon_regiones": stats.wilcoxon_paired(ra, rb),
                "wilcoxon_chips": stats.wilcoxon_paired([x["iou"] for x in filas[a]], [x["iou"] for x in filas[b]]),
                "regiones_donde_gana_b": [r for r, x, y in zip(orden, ra, rb) if y > x]}

    res["comparaciones"] = {
        "sam_menos_unet": comparar("unet", "sam"),              # principal (b - a = SAM - U-Net)
        "sam_menos_unet_tta": comparar("unet_tta", "sam_tta"),
        "sam_menos_umbral": comparar("umbral", "sam"),
        "unet_menos_umbral": comparar("umbral", "unet"),
    }
    # Analisis secundario: caida al pasar de region vista a no vista, en los 86 chips de valid (semilla 0).
    valid = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))["valid"]
    pos = {c: i for i, c in enumerate(nombres)}
    vistos = {"sam": json.loads((config.RESULTS_DIR / "hacia_060" / "paso2" / "lora_db3_otsu_seed0.json").read_text(encoding="utf-8")),
              "unet": json.loads((config.RESULTS_DIR / "hacia_060" / "paso3" / "unet_db3_seed0.json").read_text(encoding="utf-8"))}
    caida = {}
    for m in ("sam", "unet"):
        visto = vistos[m]["valid"]["individual"]["por_chip"]
        no_visto = [filas[m][pos[c]] for c in valid]
        g_v, g_n = stats.aggregate(visto)["global_iou"], stats.aggregate(no_visto)["global_iou"]
        caida[m] = {"region_vista": g_v, "region_no_vista": g_n, "caida": g_v - g_n,
                    "bootstrap": stats.paired_bootstrap(visto, no_visto, n_boot=N_BOOT)["global_iou"]}
    reg_valid = [data.region_of(c) for c in valid]
    v_sam, n_sam = vistos["sam"]["valid"]["individual"]["por_chip"], [filas["sam"][pos[c]] for c in valid]
    v_un, n_un = vistos["unet"]["valid"]["individual"]["por_chip"], [filas["unet"][pos[c]] for c in valid]
    # diferencia de caidas: (SAM no visto - SAM visto) - (U-Net no visto - U-Net visto), bootstrap por region
    rng = np.random.default_rng(0)
    arr = {k: np.array([[r["tp"], r["fp"], r["fn"]] for r in v], float) for k, v in
           (("vs", v_sam), ("ns", n_sam), ("vu", v_un), ("nu", n_un))}
    uniq = sorted(set(reg_valid))
    miembros = {g: np.nonzero(np.array(reg_valid) == g)[0] for g in uniq}

    def gi(a, i):
        t = a[i].sum(0)
        return t[0] / t.sum()

    difs = []
    for _ in range(N_BOOT):
        i = np.concatenate([miembros[g] for g in rng.choice(uniq, len(uniq))])
        difs.append((gi(arr["ns"], i) - gi(arr["vs"], i)) - (gi(arr["nu"], i) - gi(arr["vu"], i)))
    todo = np.arange(len(valid))
    caida["diferencia_de_caidas"] = {
        "valor": float((gi(arr["ns"], todo) - gi(arr["vs"], todo)) - (gi(arr["nu"], todo) - gi(arr["vu"], todo))),
        "ic95_por_region": [float(x) for x in np.percentile(difs, [2.5, 97.5])],
        "lectura": "positivo = SAM pierde menos que la U-Net al pasar a una region no vista"}
    res["caida_por_generalizacion_valid"] = caida
    res["filas_por_chip"] = {"chips": nombres, "regiones": grupos} | {m: filas[m] for m in METODOS}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: 2 regiones, pocos chips, 1-2 épocas")
    a = ap.parse_args()
    global OUT, CACHE
    config.setup_runtime(seed=SEED, verbose=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("Necesita la GPU (CUDA).")
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    chips = parts["train"] + parts["valid"]
    P = {"epochs_lora": L.EPOCHS, "epochs_unet": P3.UNET_EPOCHS}
    if a.rapido:
        import shutil

        OUT, CACHE = OUT / "prueba_rapida", OUT / "prueba_rapida" / "modelos"
        shutil.rmtree(OUT, ignore_errors=True)
        keep = [c for c in chips if data.region_of(c) in ("Nigeria", "Somalia")][:6] + \
               [c for c in chips if data.region_of(c) in ("Spain",)][:10]
        chips, P = keep, {"epochs_lora": 1, "epochs_unet": 2}
    assert not any(data.region_of(c) == "Bolivia" for c in chips)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    regiones = [data.region_of(c) for c in chips]
    orden = sorted(set(regiones))
    if a.rapido:
        orden = orden[:2]
    reg = Registro(OUT)
    reg.info(f"=== Validación por región: {len(chips)} chips, {len(set(regiones))} regiones, GPU {torch.cuda.get_device_name(0)}")
    reg.set(etapa="preparando chips", total_pliegues=len(orden))
    items = L.prepare_chips(chips, "db3", dev)
    refs = referencias_sin_entrenar(chips)
    reg.info("Chips preparados y referencias calculadas.")
    pliegues = {}
    for k, region in enumerate(orden, 1):
        f = OUT / f"pliegue_{region}.json"
        if f.exists():
            pliegues[region] = json.loads(f.read_text(encoding="utf-8"))
            reg.info(f"[{k}/{len(orden)}] {region}: ya estaba terminado, se reutiliza.")
            continue
        t0 = time.time()
        n_te = sum(r == region for r in regiones)
        reg.info(f"[{k}/{len(orden)}] {region}: {n_te} chips fuera, {len(chips) - n_te} para entrenar")
        res = correr_pliegue(region, chips, regiones, items, refs, P, reg, dev)
        save_json(res, f)
        pliegues[region] = res
        ir = res["iou_region"]
        reg.info(f"[{k}/{len(orden)}] {region} TERMINADO en {(time.time() - t0) / 60:.1f} min | IoU región no vista: "
                 f"SAM {ir['sam']:.3f} (TTA {ir['sam_tta']:.3f}) | U-Net {ir['unet']:.3f} (TTA {ir['unet_tta']:.3f}) | "
                 f"umbral {ir['umbral']:.3f} | Otsu {ir['otsu']:.3f}")
        reg.estado["pliegues_terminados"].append(region)
        reg.set(etapa=f"{len(reg.estado['pliegues_terminados'])}/{len(orden)} pliegues terminados", detalle="")
    if a.rapido:
        reg.info("Prueba rápida terminada (no se calcula el resumen).")
        return
    res = resumen(chips, regiones, pliegues)
    save_json(res, OUT / "resumen.json")
    g = res["agregado"]
    c = res["comparaciones"]["sam_menos_unet"]["por_region"]["global_iou"]
    reg.info(f"RESUMEN: IoU global (336 chips, región no vista) SAM {g['sam']['global_iou']:.4f} | U-Net "
             f"{g['unet']['global_iou']:.4f} | SAM − U-Net {c['diff_b_menos_a']:+.4f} IC por región "
             f"[{c['ic95'][0]:+.4f}, {c['ic95'][1]:+.4f}]")
    reg.set(etapa="terminado", terminado=True)


if __name__ == "__main__":
    main()
