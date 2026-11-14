# Bitácora hacia IoU 0.60

Registro de lo que probamos para pasar de **0.431** (H2) a la meta de **0.60** que pidió el profesor: qué se decidió, en qué estado está cada cosa y qué resultó. Se lee de arriba hacia abajo: tablero → pre-registro → resultados → registro cronológico.

* **Métrica (decidida por el equipo el 05-nov):** la principal es el **IoU global** de agua, igual que en H2: se suman TP, FP y FN de todos los chips antes de dividir. Además se reporta siempre el **IoU medio por chip**, con la explicación de por qué es más bajo (sección 3i).
* **`test` y `bolivia` se evaluaron una sola vez el 05-nov** (sección 3g). Todo lo que se haga después es exploratorio respecto de test.
* **Punto de partida:** Otsu con apertura 0.256 · SAM v2 0.115 · H2, decodificador ajustado, 0.431 ± 0.038.

## Cómo ver el avance mientras entrena

* **Doble clic en `tools\lanzar_hacia060.bat`.** Abre una ventana con dos barras: el **% total** (con tiempo transcurrido y restante) y la etapa actual (configuración, lr, época e IoU en train-dev). Al terminar cada época se imprime una línea con la pérdida y el IoU.
* `results/hacia_060/estado.json`: el mismo avance en texto (porcentaje, etapa, último IoU). Se actualiza cada 2 s.
* `results/hacia_060/ejecucion.log`: una línea por época, para revisar después.
* **Esta bitácora se actualiza sola** al empezar y al terminar cada configuración (tabla de la sección 3).
* Se puede cortar con Ctrl+C. Lo terminado queda guardado y, al relanzar, el script se salta las configuraciones listas.
* Prueba de humo de unos 2 minutos, que no toca ningún resultado: `tools\lanzar_hacia060.bat --rapido`.

## 1. Tablero de estrategias

| # | Estrategia | Por qué (evidencia) | Estado | Resultado |
|---|---|---|---|---|
| 0 | Investigar trabajos parecidos y diagnosticar el código | Ver sección 5 | ✅ Terminado (04-nov) | Tres problemas hallados: la normalización por chip, la entrada solo VH y los prompts de Otsu en chips sin agua |
| 1a | **Normalización con rango fijo en dB** (`vhdb3`, `db3`) | `normalize_band` estira cada chip con sus propios percentiles 2-98 y borra el valor absoluto de retrodispersión, que es justo lo que distingue el agua. Garg et al. (2023) recortan a un rango fijo en dB | ✅ Terminado (04-nov) · **sirve** | Con puntos de Otsu, `vhdb3_otsu` saca **0.472 ± 0.017** contra 0.401 de `vh3_otsu` (+0.071, mismo entrenamiento). Es el mejor resultado en valid: +0.041 sobre H2 y la mitad de su variación entre semillas. Sin prompts casi no cambia (+0.004) |
| 1b | **Entrada [VV, VH, VV−VH]** (`db3`) | Con etiquetas manuales VV rinde mejor que VH, y la combinación de las tres suele ser la mejor (Frontiers 2022). A4 solo se probó con SAM congelado | ✅ Terminado (04-nov) · ❌ **no sirve con el codificador congelado** | `db3_otsu` 0.399 ± 0.075 contra 0.472 de `vhdb3_otsu` (−0.073 y muy inestable). Sin prompts, +0.017. Se retoma solo si se ajusta el codificador (paso 2) |
| 1c | **Variante sin prompts** (`*_sin`) | En varios chips de valid sin agua real, Otsu igual marca agua, y H2 pone puntos positivos sobre tierra. SAMed entrena sin prompts | ✅ Terminado (04-nov) · mixto | Variación entre semillas mucho menor (desviación 0.009–0.017), pero media más baja que con puntos de Otsu en `vhdb3` (0.421 contra 0.472). Los puntos de Otsu todavía aportan con el codificador congelado |
| 1d | **Programa de lr coseno, elección de época y umbral en train-dev** | H2 usaba la última época, con lr constante y sobreajuste tardío (notebook 06) | ✅ Terminado (04-nov) · ❌ **no sirve tal como se diseñó** | El control `vh3_otsu` da 0.401 ± 0.047 contra 0.431 de H2. Train-dev sobrestimó valid entre 0.06 y 0.22 en todas las configuraciones, y su orden no coincide con el de valid (ver sección 3b) |
| 2 | **LoRA en el codificador + decodificador, con rotaciones y espejados al vuelo** (H5) | El codificador congelado "ve" poco el SAR (sonda 0.93 contra 0.906 de los dB crudos). Los trabajos que superan 0.55 ajustan el codificador. El paso 1 agotó lo que da el decodificador solo | ✅ Terminado (04-nov) · **sirve: cruza 0.60 en valid, por poco** | Métrica principal (regla pre-registrada): `lora_vhdb3_sin` **0.602 ± 0.004** y `lora_db3_otsu` **0.601 ± 0.010**, ambas ≥ 0.60. `lora_vhdb3_otsu` 0.596 ± 0.016. Las 9 semillas superan al paso 1 (+0.11 a +0.16) y a H2 (+0.11 a +0.22) con IC 95 % que excluyen el cero. Mejor secundaria: `lora_db3_otsu` con ensamble + TTA, 0.615. Margen mínimo: 4 de 9 semillas quedan bajo 0.60 (ver 3d) |
| 3a | **Línea base U-Net** desde cero, con los mismos chips, la misma entrada y los mismos aumentos | Comparación honesta exigida en el E3: ¿SAM aporta algo frente a una red simple? | ✅ Terminado (05-nov) · ⚠️ **supera a SAM** | **0.637 ± 0.003** (ensamble + TTA 0.642), contra 0.601 de `lora_db3_otsu`. Gana en las 3 semillas (+0.027 a +0.042, Wilcoxon p ≤ 0.035), aunque el IC por chip excluye el cero solo en 1 de 3. Es el mejor resultado de todo el trabajo, con 10 min de entrenamiento por semilla y sin preentrenamiento (sección 3f) |
| 3b | **Pérdida Tversky** (α 0.3, β 0.7) en vez de Dice, sobre `lora_db3_otsu` | El error dominante del paso 2 es el agua no detectada (sensibilidad 0.68–0.74 contra precisión 0.76–0.82) | ✅ Terminado (05-nov) · ❌ **no cumple la regla de mejora** | **0.615 ± 0.008** (ensamble + TTA 0.622), +0.014 sobre el paso 2, pero ningún IC por semilla excluye el cero (se pedían 2 de 3). La sensibilidad sube algo (0.75 contra 0.70) y la precisión baja (0.77 contra 0.81), como se esperaba. El candidato para test sigue siendo `lora_db3_otsu` con ensamble + TTA |
| 3c | Preentrenar con etiquetas débiles (4 385 chips) y ajustar con las manuales | El IoU de entrenamiento es casi igual al de valid: al modelo le falta aprendizaje, no se sobreajusta. Garg et al.: +1 a +3 puntos | ❌ Cancelado por el equipo (05-nov), antes de terminar la primera semilla | Sin resultado. Se intentó como exploración E2 (sección 2d); detalle en el registro cronológico |
| 5 | Evaluación final en `test` y `bolivia` | Reservada para el E3 | ✅ Terminado (05-nov), una sola vez · **cruza 0.60** | Candidato SAM (ensamble + TTA): **test 0.642**, **Bolivia 0.657**. U-Net (línea base): test 0.666, Bolivia 0.628. Otsu: 0.263 y 0.467. La U-Net gana en test y SAM gana en Bolivia, la región que ninguno vio al entrenar (sección 3h) |

Leyenda: ✅ terminado · ⏳ en curso · ⬜ pendiente · ❌ descartado (con motivo).

## 2. Pre-registro de H4 (paso 1)

Redactado el **4 de noviembre de 2026 a las 16:55 (hora de Lima), antes de ejecutar**. Ningún número de esta sección viene de una corrida del paso 1.

> **Declaración de IA.** Claude Code (Opus 5.5) propuso este diseño y escribió el código (`src/inf659sam/hacia060.py`, `tools/hacia060_paso1.py`) a pedido del equipo. Las expectativas razonadas son del asistente: **el equipo debe revisarlas y firmarlas** (sección 6) antes de presentarlas como propias.

**Hipótesis H4.** Dar al decodificador una entrada SAR con escala física constante entre chips (rango fijo en dB, con VV, VH y VV−VH) y quitarle la dependencia de los prompts de Otsu sube el IoU global sobre H2, sin tocar el codificador.

**Configuraciones** (representación × prompts, 6 en total):

| Representación | Qué es | Prompts |
|---|---|---|
| `db3` | [VV, VH, VV−VH], cada canal con un rango fijo en dB | `otsu` (3+2, como H2) y `sin` (sin puntos: embedding por defecto del codificador de prompts) |
| `vhdb3` | VH con rango fijo, replicada ×3 (aísla la normalización del cambio de canales) | `otsu` y `sin` |
| `vh3` | La entrada de H2 (percentiles 2-98 por chip), como control del nuevo entrenamiento | `otsu` y `sin` |

**Rangos fijos.** Los percentiles 1 y 99 de cada canal (VV, VH y VV−VH) sobre los píxeles con dato del sensor de **los 250 chips de `train`** (1 de cada 4 filas y columnas). Se guardan en `results/hacia_060/rangos_db.json`. Valid no interviene. Los píxeles sin dato se rellenan con la mediana de train de cada canal.

**Entrenamiento (igual para las 6).** Codificador de imagen y de prompts congelados, solo `mask_decoder` (4 058 340 parámetros). BCE + Dice, AdamW (wd 0.01), lote 4, salida de una máscara. **Programa:** 20 épocas, 1 de calentamiento lineal y luego coseno hasta 0. Los prompts de entrenamiento de las variantes `otsu` son los mismos de H2 (aleatorios con semilla dentro de Otsu). Las variantes `sin` entrenan también con los chips donde Otsu no marca agua.

**Selección, solo en `train-dev`.**
1. Barrido de lr {3e-5, 1e-4, 3e-4} con semilla 0, entrenando en train-core (199 chips) y midiendo el IoU global en train-dev (51 chips) al final de **cada época** (umbral 0.5).
2. Se fija la pareja (lr, época) con mayor IoU en train-dev.
3. Con ese modelo se elige el umbral de probabilidad en {0.2, 0.3, …, 0.8}. Si hay empate, gana el más cercano a 0.5.
4. Recién entonces se entrena con **semillas 0, 1 y 2 en todo `train`** (250 chips), con el mismo programa de 20 épocas detenido en la época elegida, y se evalúa en `valid` con el umbral elegido.

**Evaluación.** Idéntica a H2: los prompts deterministas v2.1 y el post-procesamiento de `SamProcessor`. Antes de entrenar, el script verifica que la evaluación nueva reproduce el IoU publicado de H2 (semilla 0, 0.4684) y se detiene si difiere en más de 0.002. Se reportan media ± desviación entre semillas, bootstrap pareado por chip y por región (5 000 remuestreos) y Wilcoxon contra H2 (misma semilla) y contra Otsu.

**Regla de decisión.** La "candidata del paso 1" es la configuración con mayor IoU en train-dev, no la mejor en valid. En valid se reportan las 6. Si ninguna supera a H2, se reporta tal cual.

**Expectativas previas (para contrastar después):**
* `db3` y `vhdb3` superan a `vh3` con el mismo tipo de prompt: la normalización por chip es el problema principal. Esperamos de +0.05 a +0.15 de IoU global sobre H2.
* `sin` ≥ `otsu`, sobre todo por menos falsos positivos en chips sin agua.
* `vh3_otsu` con el nuevo programa queda cerca de H2 (+0.00 a +0.05), pero con menos variación entre semillas.
* **No esperamos llegar a 0.60 solo con el paso 1:** el codificador sigue congelado. Lo más probable está entre 0.45 y 0.55. El paso 2 (LoRA) es el candidato a cruzar 0.60.

**Desviaciones conocidas.** El entrenamiento en GPU no es determinista bit a bit: las semillas fijan el orden de los datos y los prompts, pero no los redondeos de CUDA. Pasaba lo mismo en H2.

## 2b. Pre-registro de H5 (paso 2: LoRA)

Redactado el **4 de noviembre de 2026 a las 18:25 (hora de Lima), después de ver el paso 1 y antes de ejecutar el paso 2**. Las decisiones que dependen del paso 1 se justifican con sus resultados en valid, y eso se declara: el paso 1 ya usó valid para elegir `vhdb3`, así que las comparaciones contra `vhdb3_otsu` son internas. La confirmación independiente queda para `test` en el E3.

> **Declaración de IA.** Diseño y código (`src/inf659sam/lora_sam.py`, `tools/hacia060_paso2.py`) propuestos por Claude Code (Opus 5.5) a pedido del equipo, que debe revisar y firmar las expectativas (sección 6).

**Hipótesis H5.** Adaptar el codificador de imagen con LoRA, junto con el decodificador y con rotaciones y espejados al vuelo, sube el IoU global en valid por encima del paso 1 (0.472) y de H2 (0.431).

**Lección del paso 1 que se aplica aquí: no hay ninguna selección.** Todos los hiperparámetros se fijan de antemano, se usa la última época y el umbral es 0.5. No hay barrido, ni train-dev, ni elección de época, así que no puede haber maldición del ganador.

**Modelo.**
* SAM ViT-B con atención SDPA.
* **LoRA de rango 8 (α = 16)** en la proyección `qkv` de los 12 bloques del codificador. B arranca en cero, así que el modelo inicial es idéntico a SAM.
* Se entrenan **LoRA y el `mask_decoder` completo** (unos 4.35 M parámetros). Quedan congelados el resto del codificador, el cuello y el codificador de prompts.
* Checkpointing de gradiente sin reentrada y bf16 automático. Así cabe en los 8 GB de la RTX 3070: unos 4.3 GB con lote 4, a unos 0.26 s por imagen.

**Entrenamiento (igual para todas).** Todo `train` (250 chips), **15 épocas** (1 de calentamiento lineal y luego coseno hasta 0), **lr 1e-4** para todo lo entrenable, AdamW (wd 0.01), lote 4 y BCE + Dice. **Aumentos D4**: en cada muestra se elige al azar una rotación de 0, 90, 180 o 270° y un espejado sí o no, y se aplican igual a la imagen, la etiqueta y los puntos. Los puntos de entrenamiento se muestrean igual que en H2: 3 positivos dentro de Otsu y 2 negativos fuera, a 2 px o más del borde. **Semillas 0, 1 y 2.**

**Configuraciones** (en este orden de ejecución):

| Nombre | Entrada | Prompts | Por qué |
|---|---|---|---|
| `lora_vhdb3_otsu` | VH con rango fijo en dB | Otsu 3+2 | Continuación directa de la mejor del paso 1 |
| `lora_db3_otsu` | [VV, VH, VV−VH] con rango fijo | Otsu 3+2 | Con el codificador adaptable, VV y VV−VH podrían servir |
| `lora_vhdb3_sin` | VH con rango fijo en dB | sin prompts | Diseño sin prompts de la literatura (SAMed, aksaN000) |

**Evaluación en valid** (86 chips, mismos píxeles):
* **Métrica principal:** IoU global, media ± desviación de las 3 semillas, modelo individual, sin TTA, umbral 0.5.
* **Métricas secundarias, fijadas de antemano:** (a) con TTA D4, promediando las probabilidades de las 8 transformaciones; (b) ensamble de las 3 semillas, promediando probabilidades; (c) ensamble + TTA.
* **Pruebas:** bootstrap pareado por chip y por región (5 000 remuestreos) y Wilcoxon contra H2 (misma semilla), contra el paso 1 `vhdb3_otsu` (misma semilla) y contra Otsu.
* **Control del cambio de preprocesamiento:** el paso 2 prepara la imagen en la GPU (redimensión bilineal de torch en vez de la de PIL que usa SamProcessor). Antes de entrenar se verifica que, con LoRA en cero y el decodificador de H2, la evaluación nueva da en valid un IoU dentro de ±0.01 de 0.4684.

**Regla de decisión.** Se dice que el paso 2 "cruza 0.60" solo si **la métrica principal de alguna configuración es ≥ 0.60**. Si solo lo logra una métrica secundaria (TTA o ensamble), se reporta así, aclarando cuál. Con tres configuraciones, la mejor en valid vuelve a ser una elección mirando valid: lo que se lleva a `test` en el E3 se fija antes de abrir test.

**Expectativas previas:**
* `lora_vhdb3_otsu` sube de +0.05 a +0.15 sobre el paso 1, o sea a 0.52–0.62. Llegar a 0.60 es posible, pero no lo esperamos con seguridad.
* `lora_db3_otsu` ≥ `lora_vhdb3_otsu`: ahora el codificador puede aprender a usar VV y VV−VH, al revés que en el paso 1.
* `lora_vhdb3_sin` queda más cerca de su versión con prompts que en el paso 1, porque el codificador adaptado necesita menos ayuda.
* TTA suma de +0.01 a +0.02, y el ensamble de +0.01 a +0.03, con menos variación.

## 2c. Pre-registro del paso 3 (3a U-Net y 3b Tversky) y modelo de respaldo para test

Redactado el **5 de noviembre de 2026 a las 02:05 (hora de Lima), después de ver el paso 2 y antes de ejecutar el paso 3**. Antes de escribirlo solo se midió la velocidad de la U-Net con datos aleatorios (5.6 s por época) y se corrió una prueba de humo de 16 chips que no toca resultados.

> **Declaración de IA.** Diseño y código (`src/inf659sam/paso3.py`, `tools/hacia060_paso3.py`) propuestos por Claude Code (Opus 5.5) a pedido del equipo, que debe revisar y firmar las expectativas (sección 6).

**Modelo de respaldo para `test`, fijado ahora.** Es `lora_db3_otsu` con ensamble de 3 semillas y TTA D4: **0.615 en valid**. Si el paso 3 no cumple la regla de abajo, este es el modelo que va a `test` en el E3. Se eligió mirando valid entre 3 configuraciones y 4 modos de predicción, y así se declarará.

**Lo común a 3a y 3b.**
* La entrada es `db3`: [VV, VH, VV−VH] con rangos fijos en dB.
* Se entrena con todo `train` (250 chips), semillas 0, 1 y 2.
* Aumentos D4 al azar por muestra.
* Sin ninguna selección: se usa la última época y el umbral 0.5.
* Evaluación en valid igual que en el paso 2: modelo individual, TTA D4, ensamble de 3 semillas y ensamble + TTA.
* Bootstrap pareado por chip y por región, y Wilcoxon contra el paso 2 `lora_db3_otsu` (misma semilla), contra H2 y contra Otsu.

**3a: `unet_db3`, línea base U-Net.**
* U-Net clásica con BatchNorm, 4 bajadas y 32 canales de base (7.76 M parámetros), entrenada **desde cero**. No se descargan pesos preentrenados.
* Recibe la misma imagen uint8 de 3 canales que SAM, normalizada como (x/255 − 0.5)/0.25. Sin prompts.
* **100 épocas** (2 de calentamiento y luego coseno), AdamW con lr 1e-3 y wd 1e-4, lote 8, BCE + Dice, bf16.
* Valores estándar para una U-Net desde cero, fijados sin probar alternativas. Desde cero necesita muchas más épocas que el ajuste de SAM; con 5.6 s por época son unos 10 min por semilla.

**3b: `lora_db3_otsu_tversky`.**
* Es idéntico a `lora_db3_otsu` del paso 2: LoRA r=8, 15 épocas, lr 1e-4, lote 4 y prompts Otsu 3+2.
* **Una sola diferencia:** el término Dice de la pérdida se reemplaza por Tversky con α = 0.3 (peso de FP) y β = 0.7 (peso de FN). Dice es el caso α = β = 0.5, así que solo se mueve el equilibrio entre ambos tipos de error.
* La BCE se mantiene.

**Regla de decisión.**
* Un experimento "**mejora al paso 2**" si cumple las dos condiciones:
  * su métrica principal (media de 3 semillas, modelo individual) supera 0.601, que es `lora_db3_otsu`;
  * en al menos 2 de las 3 semillas, el IC 95 % bootstrap por chip de la diferencia con la misma semilla del paso 2 excluye el cero.
* En cualquier otro caso se reporta como "no mejora", con sus números.
* **El candidato para test cambia** a `lora_db3_otsu_tversky` con ensamble + TTA solo si 3b mejora al paso 2 según esa regla **y** su ensamble + TTA supera 0.615.
* La U-Net no compite como candidato. Es la línea base y se reporta siempre.

**Expectativas previas:**
* **U-Net:** entre 0.50 y 0.60, por debajo de `lora_db3_otsu`. Con solo 250 chips y sin preentrenamiento debería rendir menos que SAM adaptado, pero no está garantizado. Si iguala o supera a SAM, se reporta como hallazgo central del E3.
* **Tversky:**
  * la sensibilidad sube de 0.03 a 0.06 y la precisión baja algo;
  * el IoU global cambia de −0.01 a +0.02;
  * no esperamos que cumpla la regla de mejora con seguridad.

## 2d. Protocolo del bloque de robustez y exploración (después de test)

Redactado el **5 de noviembre de 2026 a las 15:00**, antes de ejecutar. **Test y Bolivia ya se usaron**, así que nada de esto cambia el candidato ni el resultado principal. Lo que se mida en test se reporta como **análisis posterior**.

**R1. Línea base clásica fuerte (umbral filtrado).** La idea sale de sar-flood-unet (sección 5c).
* VH en dB, con filtro de mediana 5×5. Los píxeles sin dato se rellenan con la mediana del chip antes de filtrar y no se puntúan.
* Umbral fijo, elegido **solo en train** (250 chips), barriendo de −40 a +5 dB en pasos de 0.1 y maximizando el IoU global.
* Se evalúa en valid, test y Bolivia, junto con la variante sin filtro.

**R2. Cinco semillas.** Se entrenan las semillas 3 y 4 de `lora_db3_otsu` y de `unet_db3`, con el mismo protocolo exacto.
* Se reportan **solo en valid**: media ± desviación de 5 semillas, ensamble de 5 + TTA y la diferencia U-Net − SAM con 5 pares de semillas.
* Pregunta: ¿la ventaja de la U-Net en valid se mantiene con 5 semillas?

**R3. Figuras finales de test y Bolivia** con las predicciones de los modelos ya fijados (no se entrena nada). Incluyen la figura del IoU por chip contra el % de agua, que explica por qué cae el IoU medio por chip. Los chips de ejemplo se eligen con una regla fija: en cada conjunto, los 2 chips con mayor diferencia a favor de cada modelo y los 2 con más agua.

**E1. U-Net con canal JRC (exploratorio).**
* Igual que `unet_db3`, pero con un 4.º canal: la capa `JRCWaterHand` de Sen1Floods11, agua permanente del JRC.
* 3 semillas, la misma receta de 100 épocas.
* Se evalúa en valid, y en test y Bolivia como exploratorio.

**E2. Preentrenar con etiquetas débiles (exploratorio, solo U-Net por costo).** Se usan los chips `S1Weak` con etiquetas `S2IndexLabelWeak`, **excluyendo los de Bolivia** para que siga siendo una región no vista. (Corrección de las 15:20, antes de correr: la primera versión de este protocolo decía que esas etiquetas no incluían Bolivia, pero al revisar los archivos sí aparecen 224 chips.) Hay dos variantes:
* **E2a:** etiquetas débiles crudas.
* **E2b:** etiquetas débiles **corregidas con la ocurrencia de agua del JRC**: se marca como agua todo píxel con ocurrencia ≥ 50 %, como en Garg et al. Se usan los mosaicos públicos de JRC Global Surface Water, recortados a cada chip.

En ambas se preentrena 10 épocas con las débiles (misma entrada `db3`, D4, BCE + Dice, lr 1e-3 con coseno) y luego se ajusta con la receta exacta de `unet_db3` sobre los 250 chips manuales. Son 3 semillas por variante.

**Expectativas.** Según Garg et al., E2a no debería mejorar (≈ +0.00) y E2b podría sumar de +0.01 a +0.03. Esperamos que E1 sume algo en chips con ríos o lagos estables (+0.00 a +0.02). En R1, el umbral filtrado debería quedar alrededor de 0.55–0.60 en test, entre Otsu y los modelos.

## 2e. Pre-registro de la validación cruzada por región (R4, exploratorio)

Redactado el **12 de noviembre de 2026 a las 15:15 (hora de Lima), antes de ejecutar**. Motivación y diagnóstico previo en `docs/siguiente_paso_sam_vs_unet.md`.

> **Declaración de IA.** Diseño y código (`tools/validacion_por_region.py`) preparados por Claude Code (Opus 5.5) a pedido del equipo, que debe revisar las expectativas.

* **Pregunta:** ¿SAM con LoRA generaliza mejor que la U-Net a regiones que no vio al entrenar?
* **Datos:** train y valid juntos, 336 chips de 10 regiones. **Test y Bolivia no se usan.**
* **Diseño:** 10 pliegues. En cada uno se entrena con 9 regiones y se predice la que quedó fuera.
* **Modelos:** `lora_db3_otsu` (paso 2) y `unet_db3` (paso 3a) con su configuración ya fijada: misma entrada, mismas épocas, misma tasa de aprendizaje, última época, umbral 0.5, **semilla 0**. Sin búsqueda de hiperparámetros.
* **Referencias sin entrenamiento:** Otsu por chip y el umbral fijo en VH con mediana 5 × 5, elegido en cada pliegue solo con las 9 regiones de entrenamiento.
* **Métrica principal:** IoU global sobre los 336 chips, cada uno predicho por el modelo que no vio su región, modelo individual sin TTA. **Secundarias:** lo mismo con TTA; IoU medio por chip; IoU global de cada región.
* **Pruebas:** diferencia SAM − U-Net con bootstrap pareado por región (principal) y por chip (5 000 remuestreos), y Wilcoxon sobre los 10 IoU de región.
* **Regla:** se dirá que un modelo generaliza mejor solo si el IC 95 % del bootstrap por región excluye el cero. Si lo incluye, se dirá que no hay evidencia de diferencia.
* **Análisis secundario fijado ahora:** caída por generalización en los 86 chips de valid, comparando el IoU con región vista (paso 2 y 3a, semilla 0: 0.612 y 0.639) contra el IoU con región no vista de este experimento. Si el preentrenamiento ayuda a generalizar, SAM debería caer menos.
* **Limitación declarada:** los rangos fijos en dB de la entrada `db3` se calcularon con todo train, así que en cada pliegue incluyen algo de la región que queda fuera. Son 3 percentiles por banda sin etiquetas, y se mantienen para no cambiar los modelos.
* **Expectativas previas:** incierta. La literatura (sección 5d y la nota) sugiere una ventaja pequeña o nula del modelo preentrenado. Esperamos que la diferencia caiga dentro de ±0.03 y que el IC por región incluya el cero; que SAM caiga algo menos que la U-Net al pasar de región vista a no vista; y que ambos modelos superen al umbral fijo por menos margen que en test.
* **Costo estimado:** unas 5 horas en la RTX 3070 (19 min de SAM y 11 de U-Net por pliegue).

## 3. Resultados del paso 1 (se actualiza solo)

<!-- RESULTADOS_PASO1:INICIO (se regenera solo; no editar a mano dentro) -->

**Avance del paso 1: 6/6 configuraciones terminadas.** ✅ Paso 1 terminado.

| Configuración | Entrada | Prompts | Estado | lr | Época | Umbral | IoU train-dev | **IoU global valid** (3 semillas) | IoU medio por chip | Δ vs H2 (misma semilla) | Δ vs Otsu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `db3_otsu` | [VV, VH, VV-VH], rangos fijos en dB | prompts Otsu 3+2 (como H2) | ✅ Terminado | 0.0003 | 13 | 0.5 | 0.622 | **0.399 ± 0.075** | 0.310 ± 0.022 | -0.032 | +0.144 |
| `db3_sin` | [VV, VH, VV-VH], rangos fijos en dB | sin prompts (embedding por defecto) | ✅ Terminado | 3e-05 | 13 | 0.4 | 0.523 | **0.439 ± 0.013** | 0.326 ± 0.007 | +0.008 | +0.183 |
| `vhdb3_otsu` | VH x3, rango fijo en dB | prompts Otsu 3+2 (como H2) | ✅ Terminado | 0.0001 | 13 | 0.5 | 0.581 | **0.472 ± 0.017** | 0.334 ± 0.011 | +0.041 | +0.216 |
| `vhdb3_sin` | VH x3, rango fijo en dB | sin prompts (embedding por defecto) | ✅ Terminado | 0.0001 | 13 | 0.5 | 0.509 | **0.421 ± 0.009** | 0.310 ± 0.008 | -0.010 | +0.166 |
| `vh3_otsu` | VH x3, percentiles 2-98 por chip (como H2) | prompts Otsu 3+2 (como H2) | ✅ Terminado | 0.0003 | 13 | 0.5 | 0.591 | **0.401 ± 0.047** | 0.309 ± 0.022 | -0.030 | +0.145 |
| `vh3_sin` | VH x3, percentiles 2-98 por chip (como H2) | sin prompts (embedding por defecto) | ✅ Terminado | 0.0001 | 8 | 0.4 | 0.488 | **0.418 ± 0.017** | 0.307 ± 0.008 | -0.013 | +0.162 |

Referencia (valid, 86 chips): Otsu con apertura **0.256**, H2 **0.431 ± 0.038**. Δ = diferencia media de IoU global entre semillas; los IC 95 % pareados están en `results/hacia_060/config_*.json`.

**Candidata del paso 1 según train-dev (regla pre-registrada): `db3_otsu`** (IoU train-dev 0.622). Se elige por train-dev, no por valid.

_Actualizado automáticamente: 2026-11-04 18:09_

<!-- RESULTADOS_PASO1:FIN -->

Detalle por configuración: `results/hacia_060/config_<nombre>.json` (curvas por época, elección, IoU por chip e IC 95 %). Figura: `results/figuras/hacia060_paso1.png`. Decodificadores: `cache/modelos/hacia_060/`.

## 3b. Interpretación del paso 1 (análisis posterior, NO pre-registrado)

Redactado el 4-nov-2026 después de ver los resultados. Todo lo que sigue es interpretación, no evidencia confirmatoria.

**1. Lo que dice la regla pre-registrada.** La candidata según train-dev es `db3_otsu`, que en valid da **0.399 ± 0.075**: no supera a H2 (diferencias por semilla de −0.074 a 0.000, todos los IC 95 % incluyen el cero). **H4, tal como se formuló, no se confirma.** Se reporta así.

**2. La mejor configuración en valid es `vhdb3_otsu`: 0.472 ± 0.017.** Supera a H2 en las semillas 1 y 2: +0.046 (Wilcoxon p = 0.016) y +0.092 (IC 95 % [+0.020, +0.160], también con bootstrap por región). En la semilla 0 queda en −0.016. Su variación entre semillas es la mitad que la de H2. **Ojo:** se eligió mirando valid, así que este 0.472 es optimista y solo `test` puede confirmarlo en el E3.

**3. Qué componente explica qué** (comparaciones con todo lo demás igual):

| Cambio | Con puntos de Otsu | Sin prompts |
|---|---|---|
| Normalización fija en dB (`vhdb3` − `vh3`) | **+0.071** | +0.004 |
| Agregar VV y VV−VH (`db3` − `vhdb3`) | −0.073 (y desviación 0.075) | +0.017 |
| Quitar los prompts (`sin` − `otsu`) | — | `vhdb3`: −0.051 · `db3`: +0.040 · `vh3`: +0.017 |
| Nuevo programa + selección en train-dev (`vh3_otsu` − H2) | −0.030 | — |

* **La normalización fija sirve.** Frente al control con el mismo entrenamiento (`vh3_otsu`), baja los falsos positivos (433 → 421 mil, 799 → 630 mil y 437 → 402 mil píxeles en las semillas 0, 1 y 2) y sobre todo el agua perdida (1.27 → 1.03 millones, 0.87 → 0.87 y 1.19 → 0.96 millones). Frente a H2, la precisión sube de 0.665 a 0.69–0.76 con una sensibilidad parecida (0.54–0.61 contra 0.555).
* **Agregar VV y VV−VH no sirve con el codificador congelado**, como ya sugería la ablación A4. El codificador entrenado con fotos RGB no sabe que esos canales son polarizaciones distintas.
* **Train-dev engañó.** Con 51 chips y 60 candidatos (3 lr × 20 épocas), elegir el máximo infla el número: es la maldición del ganador. Cinco de las seis configuraciones eligieron la época 13. Train-dev sobrestimó valid entre 0.06 y 0.22. Por eso el control `vh3_otsu` quedó peor que H2: se eligió lr = 3e-4 por un pico ruidoso.
* **Dónde se pierde IoU** (analizado en `db3_otsu` contra H2): en pocos chips muy inundados (India_1018317, con 81 % de agua, Nigeria_984831, Sri-Lanka_321316 y Mekong_98310), donde el modelo deja sin marcar agua. Como el IoU global suma píxeles, esos pocos chips pesan mucho. La sensibilidad, todavía por debajo de 0.61 en todas las configuraciones, es el cuello de botella.

**4. Distancia a la meta.** El mejor resultado (0.472) está a **0.13 de 0.60**. Con el codificador congelado, el paso 1 ya rindió casi todo lo esperable (la expectativa previa era de 0.45 a 0.55). Para cruzar 0.60 hay que adaptar el codificador (paso 2).

**5. Qué cambiar para el paso 2** (a pre-registrar antes de correrlo):
* Partir de `vhdb3` (rango fijo en dB, solo VH) y no de `db3`. Con LoRA en el codificador conviene volver a probar `db3`, porque ahí el codificador sí puede aprender a usar los canales.
* **Elegir hiperparámetros con validación cruzada de 5 particiones en train**, en vez de un solo train-dev de 51 chips, y fijar pocas opciones de antemano (por ejemplo, una sola lr y una época fija) para no repetir la maldición del ganador.
* Bajar la variación: promediar las predicciones de las 3 semillas (ensamble) o los pesos con EMA.
* Considerar una pérdida que castigue más el agua perdida (Tversky con β > α) por los chips muy inundados.

## 3c. Resultados del paso 2: LoRA (se actualiza solo)

<!-- RESULTADOS_PASO2:INICIO (se regenera solo; no editar a mano dentro) -->

**Avance del paso 2: 3/3 configuraciones terminadas.** ✅ Paso 2 terminado.

| Configuración | Entrada | Prompts | Semillas listas | **IoU global valid** (modelo individual) | Con TTA D4 | Ensamble 3 semillas | Ensamble + TTA | IoU medio por chip | Δ vs H2 | Δ vs paso 1 (`vhdb3_otsu`) |
|---|---|---|---|---|---|---|---|---|---|---|
| `lora_vhdb3_otsu` | VH x3, rango fijo en dB | prompts Otsu 3+2 | ✅ 3/3 | **0.596 ± 0.016** | 0.590 ± 0.008 | 0.599 | 0.591 | 0.385 ± 0.003 | +0.165 | +0.124 |
| `lora_db3_otsu` | [VV, VH, VV-VH], rangos fijos en dB | prompts Otsu 3+2 | ✅ 3/3 | **0.601 ± 0.010** | 0.611 ± 0.008 | 0.612 | 0.615 | 0.386 ± 0.010 | +0.170 | +0.129 |
| `lora_vhdb3_sin` | VH x3, rango fijo en dB | sin prompts | ✅ 3/3 | **0.602 ± 0.004** | 0.595 ± 0.005 | 0.603 | 0.596 | 0.377 ± 0.002 | +0.171 | +0.130 |

Referencias en valid (86 chips): Otsu **0.256** · H2 **0.431 ± 0.038** · paso 1 `vhdb3_otsu` **0.472 ± 0.017** · meta **0.60**. La métrica principal es la columna en negrita (media de 3 semillas, modelo individual, sin TTA). Las demás columnas son secundarias, fijadas de antemano.

Mejor configuración completa (métrica principal): `lora_vhdb3_sin` = 0.602 → ✅ **supera 0.60**.

_Actualizado automáticamente: 2026-11-05 15:32_

<!-- RESULTADOS_PASO2:FIN -->

Detalle: `results/hacia_060/paso2/` (un JSON por semilla con la curva por época, el IoU por chip y los IC 95 %, y un JSON por configuración con el ensamble). Avance en vivo: `results/hacia_060/paso2/estado.json` y `ejecucion.log`. Figura: `results/figuras/hacia060_paso2.png`. Pesos (LoRA + decodificador): `cache/modelos/hacia_060_paso2/`.

## 3d. Interpretación del paso 2 (análisis posterior, NO pre-registrado salvo el punto 1)

Redactado el 4-nov-2026 a las 23:45, después de ver los resultados. El punto 1 aplica la regla pre-registrada. El resto es interpretación, incluido el IC del punto 2, que se calculó después.

**1. Lo que dice la regla pre-registrada.** Dos configuraciones tienen la métrica principal en 0.60 o más: `lora_vhdb3_sin` **0.602 ± 0.004** (semillas: 0.601, 0.606, 0.599) y `lora_db3_otsu` **0.601 ± 0.010** (0.612, 0.600, 0.592). **El paso 2 cruza 0.60 en valid según la regla.** `lora_vhdb3_otsu` queda en 0.596 ± 0.016. **H5 se confirma.** Las 9 semillas superan:
* al paso 1, por +0.107 a +0.160;
* a H2, por +0.111 a +0.218.

En todas, los IC 95 % pareados (por chip y por región) excluyen el cero, y el Wilcoxon da p ≤ 0.032.

**2. Por qué hay que leerlo con cuidado.**
* **El margen es mínimo.** 4 de las 9 semillas quedan por debajo de 0.60 (0.579, 0.598, 0.592 y 0.599), y las medias pasan 0.60 por apenas 0.001–0.002.
* **Hay incertidumbre de muestreo.** Valid tiene solo 86 chips. El IC 95 % bootstrap por chip del IoU global de los ensambles va de ~0.50 a ~0.68 (`lora_db3_otsu`: [0.530, 0.680]). En otro conjunto de chips parecido, como `test`, un modelo que aquí da 0.60 puede dar 0.55 o 0.65.
* **Elegir la mejor de 3 mirando valid infla un poco el número.** Pesa menos que en el paso 1, porque aquí son 3 candidatas y no 60.
* **El preprocesamiento en GPU juega en contra, no a favor.** Con el mismo modelo, da 0.005 menos que el de SamProcessor (verificación inicial), así que no infla los números.

**3. Expectativas previas contra lo observado.**

| Expectativa (sección 2b) | Observado | ¿Se cumplió? |
|---|---|---|
| `lora_vhdb3_otsu` sube de +0.05 a +0.15 sobre el paso 1 (0.52–0.62) | +0.124 (0.596) | ✅ |
| `lora_db3_otsu` ≥ `lora_vhdb3_otsu` | 0.601 contra 0.596 (dentro del ruido) | ✅ por poco. El signo se invierte respecto al paso 1 (−0.073): con el codificador adaptable, VV y VV−VH ya no estorban |
| `lora_vhdb3_sin` más cerca de su versión con prompts | 0.602 contra 0.596: igual o mejor, con la menor variación (0.004) | ✅ |
| La TTA suma de +0.01 a +0.02 | +0.010 en `db3_otsu`; −0.006 y −0.007 en las de VH | ❌ solo en `db3` |
| El ensamble suma de +0.01 a +0.03 | De +0.001 a +0.011 | ❌ menos de lo esperado |

**4. Qué cambió respecto al paso 1.**
* La sensibilidad subió de 0.54–0.61 a **0.68–0.74**, con una precisión parecida o mejor (0.76–0.82). El codificador adaptado recupera buena parte del agua que antes se perdía.
* LoRA más los aumentos aportan unos +0.12 sobre el mejor decodificador solo.

**5. Dónde sigue perdiendo** (ensambles de 3 semillas):
* **Paraguay_581976 concentra entre el 6 y el 10 % de todo el error**, y casi todo es agua no detectada: de 62 a 109 mil píxeles. Ghana_1033830 y Nigeria_984831 son parecidos: zonas grandes de agua que quedan sin marcar.
* **Somalia_12849 y Somalia_649376 son el caso contrario.** Tienen falsos positivos (de 40 a 72 mil píxeles) en chips con poca agua. `db3` los reduce bastante: en Somalia_12849 baja de ~71 mil a 18 mil.
* **El IoU medio por chip es solo 0.38**, aunque H2 tenía 0.31.
  * 29 de los 86 chips tienen menos del 1 % de agua. En ellos el IoU medio es 0.13–0.15, porque unos pocos píxeles de error hunden el IoU de un chip casi seco.
  * En los otros 57 chips, el IoU medio es ~0.50.
  * **Si el profesor mide el 0.60 como IoU medio por chip, estamos lejos. Con IoU global, estamos justo en la meta.**

**6. Siguiente paso (pre-registrar antes de correr):**
* **Confirmar con el profesor la métrica (global o media por chip) y el conjunto (valid o test).** Cambia por completo la lectura de este resultado.
* **Fijar qué modelo va a `test` en el E3 antes de abrirlo.** El equipo decide.
  * La propuesta del asistente es `lora_db3_otsu` con ensamble de 3 semillas y TTA (0.615 en valid), que es el valor más alto y tiene la menor diferencia entre métricas.
  * La alternativa más simple es `lora_vhdb3_sin`, que no depende de Otsu y tiene la menor variación.
* **Ganar margen con el paso 3.** Las opciones son:
  * una pérdida Tversky que castigue más el agua perdida, que es el error dominante;
  * más épocas o un rango de LoRA mayor;
  * preentrenar con las etiquetas débiles.

## 3e. Resultados del paso 3: U-Net y Tversky (se actualiza solo)

<!-- RESULTADOS_PASO3:INICIO (se regenera solo; no editar a mano dentro) -->

**Avance del paso 3: 2/2 experimentos terminados.** ✅ Terminado.

| Paso | Experimento | Qué es | Semillas | **IoU global valid** (individual) | Con TTA | Ensamble | Ensamble + TTA | IoU medio por chip | Precisión | Sensibilidad | Δ vs paso 2 `lora_db3_otsu` (misma semilla) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3a | `unet_db3` | U-Net desde cero (base 32), [VV, VH, VV−VH] con rango fijo, BCE + Dice | ✅ 3/3 | **0.637 ± 0.003** | 0.639 ± 0.004 | 0.641 | 0.642 | 0.413 ± 0.005 | 0.795 ± 0.012 | 0.764 ± 0.015 | +0.036 |
| 3b | `lora_db3_otsu_tversky` | `lora_db3_otsu` con BCE + Tversky (α 0.3, β 0.7) en vez de BCE + Dice | ✅ 3/3 | **0.615 ± 0.008** | 0.620 ± 0.010 | 0.619 | 0.622 | 0.399 ± 0.001 | 0.773 ± 0.006 | 0.751 ± 0.012 | +0.014 |

Referencias en valid (86 chips): H2 **0.431 ± 0.038** · paso 2 `lora_db3_otsu` **0.601 ± 0.010** (TTA 0.611, ensamble 0.612, ensamble + TTA 0.615) · meta **0.60**. Métrica principal en negrita: media de 3 semillas, modelo individual, sin TTA, umbral 0.5.

_Actualizado automáticamente: 2026-11-05 16:00_

<!-- RESULTADOS_PASO3:FIN -->

Detalle: `results/hacia_060/paso3/`, con un JSON por semilla y uno por experimento con el ensamble. Avance en vivo: `results/hacia_060/paso3/estado.json` y `ejecucion.log`. Pesos y probabilidades: `cache/modelos/hacia_060_paso3/`. Para lanzarlo: doble clic en `tools\lanzar_hacia060_paso3.bat`.

## 3f. Interpretación del paso 3 (análisis posterior, NO pre-registrado salvo el punto 1)

Redactado el 5-nov-2026 a las 14:00, después de ver los resultados.

**1. Lo que dice la regla pre-registrada (sección 2c).**

* **3b, Tversky: no mejora al paso 2.** La media es 0.615 ± 0.008, por encima de 0.601, pero los IC 95 % por chip contra la misma semilla del paso 2 incluyen el cero en las 3 semillas: [−0.022, +0.033], [−0.020, +0.035] y [−0.013, +0.091]. Se pedían 2 de 3. **El candidato para test sigue siendo `lora_db3_otsu` con ensamble + TTA (0.615).**
* **3a, U-Net: es la línea base y se reporta siempre.** Da **0.637 ± 0.003** (semillas 0.639, 0.639 y 0.634). Con ensamble da 0.641, y con ensamble + TTA, **0.642**. La U-Net no compite como candidato, según lo pre-registrado.

**2. El hallazgo central: una U-Net simple, desde cero, rinde más que SAM adaptado.**

| Comparación de la U-Net contra el paso 2 `lora_db3_otsu` | Diferencia | IC 95 % por chip | Wilcoxon (chips mejor / peor) |
|---|---|---|---|
| Semilla 0 | +0.027 | [−0.008, +0.064] | p = 0.001 (52/24) |
| Semilla 1 | +0.039 | [+0.008, +0.071] | p < 0.001 (58/18) |
| Semilla 2 | +0.042 | [−0.011, +0.111] | p = 0.035 (47/28) |
| Ensamble + TTA de cada uno | +0.028 | [−0.003, +0.062] | p = 0.002 (48/27) |

* La U-Net gana en las 3 semillas y en el ensamble, y mejora más de 0.01 de IoU en 45 chips contra 18 donde empeora. Con 86 chips, la diferencia queda cerca del límite de lo que valid puede distinguir. **Lo que sí es claro es que SAM adaptado no supera a la U-Net.**
* La U-Net también tiene el mejor IoU medio por chip (0.413 contra 0.386) y la menor variación entre semillas (0.003).
* Cuesta mucho menos: 7.8 M parámetros, 10 min por semilla y nada preentrenado. SAM ViT-B tiene 94 M parámetros y tarda 15 min por semilla.
* **La expectativa previa (0.50–0.60) se equivocó.** No anticipamos que la normalización fija en dB y los aumentos D4 bastaran para que una red pequeña aprendiera bien con solo 250 chips.

**3. Por qué SAM no aporta aquí (hipótesis, no probadas).**
* **El preentrenamiento de SAM viene de fotos RGB.** Lo que necesita este problema es la intensidad física de retrodispersión, no bordes ni objetos. La sonda lineal de la fase A ya lo sugería: el codificador aporta poco sobre los dB crudos (AUC 0.934 contra 0.906).
* **SAM predice en 256×256 y después agranda la máscara.** La U-Net trabaja a la resolución completa de 512, y eso cuenta en ríos finos y bordes.
* **LoRA r=8 deja casi todo el codificador congelado.** En entrenamiento, SAM llega a un IoU de 0.58–0.60 y la U-Net, a 0.60–0.62.
* **Dónde gana cada uno.** La U-Net gana sobre todo en chips de Paraguay con agua dispersa: Paraguay_934240 pasa de 0.00 a 0.23, y Paraguay_305760 de 0.15 a 0.42. SAM gana en algunos chips de agua compacta, como Ghana_895194, Sri-Lanka_612594 y Ghana_1033830.

**4. Qué significa para el trabajo y el E3.**
* **La meta de 0.60 en IoU global se cruza con holgura con la U-Net.** Con SAM se cruza por poco. Ojo: el IC por chip del ensamble de la U-Net sigue siendo ancho, [0.555, 0.711].
* **La pregunta del curso, "¿sirve SAM para segmentar agua en SAR?", tiene ahora una respuesta más matizada.**
  * SAM sin ajustar no sirve (H1).
  * SAM con el decodificador ajustado mejora a Otsu (H2).
  * SAM con LoRA cruza 0.60 (H5).
  * Pero una U-Net simple con la misma entrada lo iguala o lo supera.
  * Es un resultado negativo honesto y defendible, y conviene presentarlo así.
* **Sobre test (lo decide el equipo antes de abrirlo):**
  * Lo pre-registrado es evaluar el candidato SAM (`lora_db3_otsu`, ensamble + TTA).
  * **El asistente propone evaluar también la U-Net como línea base** (`unet_db3`, ensamble + TTA), fijándola ahora. Una línea base se evalúa en test para comparar, no para elegir, así que no viola lo pre-registrado. **Decidido el 05-nov a las 14:15, antes de abrir test:** se evalúan ambos, el candidato SAM (métrica principal) y la U-Net como línea base.
* **3c (etiquetas débiles):** si se corre, conviene probarla en la U-Net y en SAM por igual. Con 4 385 chips más, la ventaja de cualquiera de los dos podría cambiar.

## 3g. Evaluación final en test y Bolivia (se actualiza sola al correrla)

**Estado: ✅ evaluado una sola vez el 05-nov a las 14:30**, con SAM y la U-Net. Resultados en la tabla de abajo y la interpretación en la sección 3h. El script es `tools/evaluar_test.py` (o `tools\lanzar_evaluacion_test.bat`).

**Qué evalúa.** No entrena ni elige nada.
* **Métrica principal:** el candidato SAM pre-registrado, `lora_db3_otsu` con ensamble de 3 semillas + TTA, umbral 0.5.
* **Secundaria:** las 3 semillas por separado.
* **Referencia:** Otsu, siempre.
* **La U-Net como línea base** solo si se agrega `--modelos sam,unet`, que sigue en revisión del equipo.
* Se aplica la misma exclusión que en train y valid: chips con más del 95 % sin etiqueta.

**Cómo se usa.**
1. **Prueba segura (no toca test):** `tools\lanzar_evaluacion_test.bat` sin argumentos. Corre todo sobre valid y comprueba que reproduce los números conocidos. **Ya se hizo el 5-nov:** SAM 0.6145 y U-Net 0.6422, idénticos a los del paso 2 y el paso 3a.
2. **Evaluación real, una sola vez:** `tools\lanzar_evaluacion_test.bat --abrir-test`, o con `--modelos sam,unet`.
   * Si faltan, descarga los chips de test (90) y Bolivia (15) del bucket público de Sen1Floods11: unos 260 MB, verificados con SHA-256.
   * Tarda unos 5 minutos.
   * Si un conjunto ya fue evaluado, el script se niega a repetirlo. `--rehacer` lo permite, pero queda anotado en el JSON.

**Antes de correrlo, el equipo debe:** decidir si entra la U-Net, firmar los pre-registros (sección 6) y, de preferencia, tener la respuesta del profesor sobre la métrica.

<!-- RESULTADOS_TEST:INICIO (se regenera solo; no editar a mano dentro) -->

**Evaluación final.** Resultados abajo; se miran una sola vez.

| Conjunto | Chips | Otsu | **SAM `lora_db3_otsu`** (ensamble + TTA) | SAM, semillas por separado | U-Net `unet_db3` (ensamble + TTA) | U-Net − SAM (IC 95 % por chip) |
|---|---|---|---|---|---|---|
| `test` | 89 | 0.263 | **0.642 (medio por chip 0.405)** | 0.612 ± 0.051 | 0.666 (medio por chip 0.421) | +0.024 [-0.007, +0.061] |
| `bolivia` | 15 | 0.467 | **0.657 (medio por chip 0.438)** | 0.663 ± 0.016 | 0.628 (medio por chip 0.397) | -0.029 [-0.069, +0.021] |

Métrica principal: IoU global del candidato SAM pre-registrado (sección 2c). La U-Net es línea base: se compara, no se elige.

_Actualizado automáticamente: 2026-11-05 14:24_

<!-- RESULTADOS_TEST:FIN -->

## 3h. Interpretación de test y Bolivia (análisis posterior; la regla aplicada es la de la sección 2c)

Redactado el 5-nov-2026 a las 14:40, después de ver los resultados.

**1. Resultado principal (pre-registrado).** El candidato SAM `lora_db3_otsu` (ensamble de 3 semillas + TTA, umbral 0.5) da:
* **test: 0.642** (89 chips; IC 95 % bootstrap por chip [0.510, 0.748]);
* **Bolivia: 0.657** (15 chips; [0.477, 0.784]).

**La meta de 0.60 en IoU global se cruza en los dos conjuntos reservados**, que no se usaron para ninguna decisión. Frente a Otsu, la mejora es de +0.379 en test ([+0.259, +0.497]) y de +0.190 en Bolivia ([+0.025, +0.350]). El IoU medio por chip queda en 0.405 y 0.438.

**2. U-Net (línea base) contra SAM.**

| | Test (89 chips, 10 regiones) | Bolivia (15 chips, región no vista) |
|---|---|---|
| SAM `lora_db3_otsu`, ensamble + TTA | 0.642 | **0.657** |
| U-Net `unet_db3`, ensamble + TTA | **0.666** | 0.628 |
| U-Net − SAM, IC 95 % por chip | +0.024 [−0.007, +0.061] | −0.029 [−0.069, +0.021] |
| Wilcoxon (chips donde gana la U-Net / gana SAM) | p = 0.011 (50 / 29) | p = 0.046 (3 / 10) |

* **En test, la U-Net vuelve a ganar**, igual que en valid. El IC por chip toca el cero, pero el bootstrap por región ([+0.002, +0.042]) y el Wilcoxon no.
* **En Bolivia, el orden se invierte y SAM gana.** Bolivia es la única región que ningún modelo vio al entrenar. Eso concuerda con la hipótesis de que el preentrenamiento de SAM generaliza mejor fuera de las regiones de entrenamiento.
  * Son solo 15 chips de una sola región, y el IC por chip incluye el cero, así que es una señal, no una prueba.
  * Con una sola región, el bootstrap por región no aplica.
* **Conclusión honesta:** dentro de las regiones de entrenamiento, una U-Net simple iguala o supera a SAM adaptado. Fuera de ellas, SAM parece generalizar algo mejor. Las dos cosas valen como resultado del trabajo.

**3. Estabilidad.** En test, las semillas de SAM por separado dan 0.642, **0.553** y 0.642, con media 0.612 ± 0.051.
* La semilla 1 falla sobre todo en Sri-Lanka_534068: deja sin marcar 191 mil píxeles de agua.
* El ensamble de 3 semillas corrige ese fallo, y por eso se fijó como candidato.
* La U-Net es mucho más estable: 0.659, 0.660 y 0.665.

**4. Desviación declarada.** El chip de test Paraguay_34417 no tiene ningún dato del radar, pero sí 50 026 píxeles de agua etiquetada.
* Se evaluó con la máscara vacía para todos los métodos (registro de las 14:25).
* Sin ese chip, SAM da 0.653 y la U-Net 0.677. La decisión no cambia ninguna conclusión.

**5. Comparación con la literatura.** Garg et al. (2023) reportan 0.676 en test con DeepLabv3+ ajustado y solo S1. Nuestra U-Net (0.666) y SAM (0.642) quedan en el mismo rango, con solo 250 chips de entrenamiento y sin etiquetas débiles.

**6. Qué ya no se puede hacer.** Test y Bolivia ya se miraron. Cualquier cambio posterior a los modelos (por ejemplo, la 3c con etiquetas débiles) ya no tendría una evaluación limpia en estos conjuntos. Si se corre, debe reportarse como exploratorio.

**Curiosidad: comparación con el paper original del dataset** (Bonafilia et al., 2020, CVPRW). El paper reporta el **IoU medio por chip** ("we report the mean IOU (equal weighting of all chips)", sección 3.2), no el global. Comparando con esa métrica, en la clase "toda el agua":

| | Test | Bolivia |
|---|---|---|
| Paper: Otsu sobre VH | 0.359 | 0.386 |
| Paper: FCNN ResNet50 con etiquetas manuales (los mismos datos que nosotros) | 0.313 | 0.352 |
| Paper: mejor modelo, con 4 385 chips de etiquetas débiles | 0.408 (débiles S2) | 0.387 (débiles S1) |
| **Nuestro SAM con LoRA** (ensamble + TTA) | 0.405 | **0.438** |
| **Nuestra U-Net** (ensamble + TTA) | **0.421** | 0.397 |

Con la métrica del propio paper, superamos a su modelo con etiquetas manuales (+0.09 a +0.11 en test) e igualamos o superamos a su mejor modelo, que usó 17 veces más datos.

Hay dos salvedades. El paper describe una partición aleatoria 60-20-20, y nosotros usamos los CSV oficiales publicados. Además, no sabemos cómo trató los chips con IoU indefinido (sin agua real ni predicha); nosotros los dejamos fuera del promedio.

Si la meta del profesor fuera 0.60 en IoU medio por chip, estaría muy por encima del paper original (0.41).

## 3i. Por qué cae el IoU medio por chip (análisis posterior)

Reportamos el **IoU global** como métrica principal y el **IoU medio por chip** como complementaria. Con el mismo modelo, en test, SAM da 0.642 global y 0.405 medio. La diferencia no es un error: las dos métricas pesan distinto a los chips.

* **El IoU global** suma los píxeles de todos los chips, así que pesan más los chips con mucha agua.
* **El IoU medio** promedia un IoU por chip, así que un chip casi seco pesa lo mismo que uno inundado.

| Test, SAM + LoRA (ensamble + TTA): agua real en el chip | Chips (% del total) | IoU medio del grupo | % de los píxeles que pesan en el IoU global |
|---|---|---|---|
| < 1 % | 28 (31 %) | **0.150** | **3.3 %** |
| 1–10 % | 37 (42 %) | 0.410 | 15.1 % |
| 10–30 % | 10 (11 %) | 0.675 | 15.1 % |
| > 30 % | 14 (16 %) | 0.670 | 66.5 % |

* **Los chips con menos de 1 % de agua son casi un tercio de los chips, pero solo el 3 % de los píxeles.** En ellos, pocos píxeles mal marcados dejan el IoU cerca de 0, y eso hunde el promedio por chip casi sin mover el global.
* A la inversa, 14 chips con más de 30 % de agua aportan dos tercios de los píxeles y concentran el resultado global.
* En Bolivia pasa lo mismo: los 3 chips con menos de 1 % de agua tienen un IoU de ~0.00 y pesan 0.6 % de los píxeles, mientras que los 4 chips con más de 30 % pesan 71 %.
* Dos chips de test sin agua real ni predicha tienen IoU indefinido y no entran al promedio.

**Lectura práctica:** el modelo mapea bien el área inundada (lo que mide el global), pero todavía comete errores relativamente grandes en imágenes casi secas (lo que castiga el medio). Las dos cosas son ciertas. Figura: `results/figuras/hacia060_iou_vs_agua.png`. Tabla completa: `results/hacia_060/robustez/iou_por_cantidad_de_agua.json`.

## 3j. Resultados del bloque de robustez y exploración (sección 2d)

**R1, línea base clásica fuerte (umbral fijo en VH elegido solo en train).** El barrido eligió −23.6 dB sin filtro y −23.3 dB con mediana 5×5, los mismos valores que sar-flood-unet.

| IoU global (IoU medio por chip) | Valid | Test | Bolivia |
|---|---|---|---|
| Otsu por chip | 0.256 | 0.263 (0.235) | 0.467 (0.427) |
| Umbral fijo, sin filtro | 0.492 (0.278) | 0.524 (0.279) | 0.531 (0.322) |
| **Umbral fijo + mediana 5×5** | 0.539 (0.306) | **0.582** (0.314) | **0.613** (0.371) |
| SAM + LoRA, ensamble + TTA | 0.615 | **0.642** (0.405) | **0.657** (0.438) |
| U-Net, ensamble + TTA | 0.642 | **0.666** (0.421) | **0.628** (0.397) |

* **En test, los dos modelos superan al umbral filtrado con claridad:** SAM +0.059 [+0.026, +0.101] y U-Net +0.083 [+0.053, +0.120].
* **En Bolivia, la ventaja es menor y ya no es concluyente:** SAM +0.044 [−0.005, +0.092] y U-Net +0.015 [−0.025, +0.057]. En una región no vista, un método clásico bien calibrado casi alcanza a los modelos.
* **La referencia honesta para el E3 es este umbral, no Otsu.** Otsu por chip sobreestima mucho la mejora de los modelos. Lo decidimos después de ver test y se declara así.

**R2, cinco semillas (terminado el 05-nov a las 16:10, solo valid).** Detalle en `results/hacia_060/robustez/cinco_semillas.json`.

| IoU global en valid | Semillas 0 a 4 | Media ± desviación | Ensamble de 5 + TTA | IoU medio por chip |
|---|---|---|---|---|
| SAM `lora_db3_otsu` | 0.612 · 0.600 · 0.592 · 0.584 · 0.625 | **0.603 ± 0.017** | 0.616 | 0.387 |
| U-Net `unet_db3` | 0.639 · 0.639 · 0.634 · 0.631 · 0.632 | **0.635 ± 0.004** | 0.643 | 0.414 |

* **La U-Net gana en las 5 semillas** (+0.006 a +0.048), pero el IC por chip excluye el cero solo en 1 de 5. Con el ensamble + TTA, la diferencia es +0.027, con IC por chip [−0.002, +0.059], IC por región [+0.004, +0.059] y Wilcoxon p = 0.002 (la U-Net gana en 50 chips y SAM en 24).
* **Lectura:** la ventaja de la U-Net se mantiene con 5 semillas, aunque la diferencia sigue en el límite de lo que 86 chips permiten afirmar.
* **Lo más claro es la estabilidad.** La desviación de SAM es unas 4 veces la de la U-Net. Con 5 semillas, SAM queda bajo 0.60 en 2 de ellas (0.592 y 0.584), mientras que la U-Net nunca baja de 0.631. Por eso el ensamble importa tanto en SAM.

**E1, U-Net con canal JRC (terminado a las 16:45).** En test y Bolivia es exploratorio.

| `unet_db3_jrc` contra `unet_db3`, ensamble + TTA | IoU global | Diferencia (IC 95 % por chip) | Sensibilidad (con JRC / sin JRC) |
|---|---|---|---|
| Valid | 0.652 | +0.010 [−0.002, +0.022] | — |
| Test | 0.673 | +0.007 [−0.000, +0.016] | — |
| **Bolivia** | **0.518** | **−0.111 [−0.187, −0.046]** | **0.568 / 0.708** |

* **Hallazgo negativo, y es el dato más útil de E1.** El canal JRC apenas ayuda en las regiones de entrenamiento, pero en la región no vista hace caer el IoU 0.11, casi todo por agua no detectada.
* **Hipótesis, no probada:** el modelo aprende a fiarse del mapa de agua permanente y deja de marcar la inundación donde el mapa no indica agua. Eso es justo lo que no se quiere en un detector de inundaciones.
* **Recomendación:** no usar la capa JRC como entrada.

**E1 y E2, exploratorio.** Esta tabla se generaba sola mientras corría la cola. Quedó fija al cancelarse E2, y su estado se corrigió a mano el 05-nov a las 21:45:

<!-- RESULTADOS_EXPLORACION:INICIO (se regenera solo; no editar a mano dentro) -->

**Exploratorio: test y Bolivia ya se habían usado.** Referencia `unet_db3`: valid 0.637 ± 0.003 (ensamble + TTA 0.642), test 0.666, Bolivia 0.628.

| Variante | Qué es | IoU global valid (3 semillas) | Ensamble + TTA valid | Δ vs `unet_db3` valid (IC 95 %) | Test, ensamble + TTA | Bolivia, ensamble + TTA | IoU medio valid |
|---|---|---|---|---|---|---|---|
| `unet_db3_jrc` | E1: U-Net con 4.º canal de agua permanente JRC | **0.648 ± 0.007** | 0.652 | +0.010 [-0.002, +0.022] | 0.673 | 0.517 | 0.414 |
| `unet_db3_debil` | E2a: preentrenada con etiquetas débiles crudas | ❌ Cancelada, sin resultado | | | | | |
| `unet_db3_debil_jrc` | E2b: preentrenada con débiles corregidas con JRC | ❌ Cancelada, nunca empezó | | | | | |

_Actualizado automáticamente: 2026-11-05 18:23_

<!-- RESULTADOS_EXPLORACION:FIN -->

## 3k. Resultados de la validación cruzada por región (R4, pre-registrada en la sección 2e)

Terminó el **13-nov-2026 a las 13:58**: 10 de 10 pliegues, código 0, solo train + valid y semilla 0. Corrió en tres tramos (12-nov 15:35–17:30, 12-nov 21:13–22:19 y 13-nov 11:08–13:58) porque el equipo la pausó dos veces. Cada pliegue se guarda completo y se reutiliza, así que las pausas no cambian nada; Nigeria, cortada a mitad en la primera pausa, se repitió entera. Redactado el 13-nov después de ver los resultados: el punto 1 aplica la regla y el resto es interpretación. Datos en `results/hacia_060/por_region/resumen.json` y figura en `results/figuras/validacion_por_region.png` (`tools/figura_por_region.py`).

| IoU global, región no vista | Chips | SAM + LoRA | U-Net | SAM − U-Net | Umbral fijo VH | Otsu |
|---|---|---|---|---|---|---|
| Ghana | 37 | 0.495 | 0.502 | −0.007 | 0.343 | 0.104 |
| India | 54 | 0.409 | 0.453 | −0.045 | 0.384 | 0.226 |
| Mekong | 24 | 0.864 | 0.848 | +0.016 | 0.781 | 0.580 |
| Nigeria | 14 | 0.634 | 0.747 | −0.113 | 0.534 | 0.333 |
| Pakistan | 22 | 0.363 | 0.310 | +0.053 | 0.255 | 0.132 |
| Paraguay | 53 | 0.565 | 0.570 | −0.005 | 0.531 | 0.243 |
| Somalia | 20 | 0.189 | 0.207 | −0.018 | 0.125 | 0.094 |
| Spain | 24 | 0.702 | 0.714 | −0.012 | 0.665 | 0.278 |
| Sri-Lanka | 33 | 0.325 | 0.510 | −0.185 | **0.597** | 0.193 |
| USA | 55 | 0.572 | 0.474 | +0.098 | 0.520 | 0.139 |
| **Total (336 chips)** | 336 | **0.554** | **0.571** | **−0.017** | 0.512 | 0.228 |
| Con TTA | | 0.556 | 0.573 | −0.018 | | |
| IoU medio por chip | | 0.344 | 0.347 | −0.003 | 0.292 | 0.223 |

**1. Lo que dice la regla pre-registrada.** SAM − U-Net = **−0.017, IC 95 % por región [−0.067, +0.029]** (por chip [−0.072, +0.024]; Wilcoxon sobre las 10 regiones p = 0.43, la U-Net gana en 7 y SAM en 3). **El IC incluye el cero: no hay evidencia de que un modelo generalice mejor que el otro a regiones no vistas.** Con TTA da lo mismo (−0.018 [−0.064, +0.028]).

**2. Expectativas previas contra lo observado.**

| Expectativa (sección 2e) | Observado | ¿Se cumplió? |
|---|---|---|
| Diferencia dentro de ±0.03 y el IC incluye el cero | −0.017, IC [−0.067, +0.029] | Sí |
| SAM cae algo menos que la U-Net al pasar de región vista a no vista (86 chips de valid) | SAM 0.612 → 0.587 (−0.025); U-Net 0.639 → 0.584 (−0.055). Diferencia de caídas +0.030, IC por región [−0.001, +0.075] | En la dirección esperada, pero el IC roza el cero: no concluyente |
| Ambos superan al umbral fijo, por menos margen que en test | SAM +0.042 [−0.037, +0.101]; U-Net +0.059 [+0.011, +0.109]. En test eran +0.059 y +0.083 | Sí, con menos margen; la ventaja de SAM sobre el umbral ya no es concluyente |

**3. Interpretación (no pre-registrada).**
* **La región pesa mucho más que el modelo.** El IoU va de 0.19 (Somalia) a 0.86 (Mekong) con los dos modelos, mientras que la diferencia entre ellos casi nunca pasa de 0.05. Las dos excepciones grandes son Sri-Lanka (−0.185, donde el umbral fijo supera a ambos) y USA (+0.098 a favor de SAM).
* **En los chips de valid, la ventaja de la U-Net desaparece fuera de la región.** Con región vista, la U-Net le ganaba a SAM por 0.027. Con región no vista quedan empatados (0.587 contra 0.584). Esto encaja con Bolivia (SAM 0.657, U-Net 0.628), aunque ninguna de las dos señales es concluyente por sí sola.
* **Conclusión para el informe:** la U-Net gana dentro de las regiones de entrenamiento. Fuera de ellas, los dos modelos son indistinguibles con estos datos, y hay un indicio débil de que SAM pierde menos al cambiar de región. Más parámetros preentrenados no dan una ventaja medible en Sen1Floods11 con Sentinel-1.
* **Límites:** una sola semilla (en valid, SAM varía ±0.017 entre semillas, así que las diferencias de una región pueden ser ruido de semilla); regiones de 14 a 55 chips; los rangos de dB de la entrada incluyen algo de cada región que queda fuera (sección 2e).

## 4. Registro cronológico

| Fecha | Qué se hizo | Resultado / decisión |
|---|---|---|
| 2026-11-04 | Investigación de trabajos parecidos y diagnóstico del código (sección 5) | Se priorizaron la normalización, los canales, el modo sin prompts y la selección en train-dev (paso 1) antes que LoRA (paso 2) |
| 2026-11-04 | Implementación del paso 1: modos `vhdb3` y `db3` en `SamWrapper.sar_to_rgb`, `src/inf659sam/hacia060.py`, lanzador con barra de avance | Pre-registro H4 escrito antes de ejecutar |
| 2026-11-04, 16:56 | Inicio de la corrida. La evaluación nueva reproduce H2: 0.46845 contra 0.46845 publicado. Rangos de train: VV [−23.9, −2.4], VH [−34.2, −9.5], VV−VH [−0.3, 17.0] dB | Verificación aprobada |
| 2026-11-04, 18:10 | Paso 1 terminado (74 min en la RTX 3070) | H4 no se confirma con la regla pre-registrada (`db3_otsu` 0.399). Mejor en valid: `vhdb3_otsu` 0.472 ± 0.017 (+0.041 sobre H2). La normalización fija sirve; VV/VV−VH y la selección en train-dev no. Siguiente: paso 2 (LoRA) partiendo de `vhdb3` |
| 2026-11-04, 18:25 | Pre-registro de H5 (sección 2b). Medición previa en la RTX 3070: LoRA con SDPA, checkpointing y lote 4 usa 4.3 GB a 0.26 s por imagen. Con checkpointing con reentrada, los gradientes no llegaban a LoRA: se usa sin reentrada | Hiperparámetros fijados sin selección |
| 2026-11-04, 18:28 | Inicio del paso 2. Verificación del preprocesamiento en GPU: 0.4635 contra 0.4684 de H2 (−0.005, dentro de ±0.01). GPU al 96–100 %, 65 s por época | 3 configuraciones × 3 semillas, unas 3 h |
| 2026-11-04, 21:03 | Paso 2 terminado: 155 min, 9 entrenamientos de 15.6 min cada uno | H5 se confirma. Con la métrica principal cruzan 0.60 `lora_vhdb3_sin` (0.602 ± 0.004) y `lora_db3_otsu` (0.601 ± 0.010). La mejor métrica secundaria es `lora_db3_otsu` con ensamble + TTA (0.615). El margen es mínimo: 4 de 9 semillas quedan bajo 0.60 y el IC 95 % en valid va de ~0.50 a ~0.68 (sección 3d). Siguiente: confirmar la métrica con el profesor y fijar el modelo para test |
| 2026-11-05, 00:30 | Figuras cualitativas con las predicciones de H2, paso 1 y paso 2 (`tools/figuras_hacia060.py`) | Se ve que el error dominante es el agua no detectada |
| 2026-11-05, 02:05 | Pre-registro del paso 3 (sección 2c). Queda fijado el modelo de respaldo para test: `lora_db3_otsu` con ensamble + TTA (0.615) | Sigue 3a (U-Net) y 3b (Tversky), unos 90 min en total |
| 2026-11-05, 02:09 | Inicio del paso 3 (`tools\lanzar_hacia060_paso3.bat`) | Primero la U-Net (3 × ~10 min) y luego Tversky (3 × ~17 min) |
| 2026-11-05, 03:29 | Paso 3 terminado (80 min) | La U-Net (línea base) da **0.637 ± 0.003**, por encima de SAM con LoRA (0.601). Tversky da 0.615 ± 0.008, pero no cumple la regla de mejora. El candidato SAM para test no cambia (`lora_db3_otsu`, ensamble + TTA). Interpretación en la sección 3f |
| 2026-11-05, 14:10 | Script de evaluación final (`tools/evaluar_test.py`), probado solo sobre valid | Reproduce valid exactamente (SAM 0.6145, U-Net 0.6422). Test y Bolivia siguen sin abrirse. La inclusión de la U-Net en test queda en revisión del equipo |
| 2026-11-05, 14:15 | Decisión, antes de abrir test: se evalúan el candidato SAM (`lora_db3_otsu`, ensamble + TTA, métrica principal) y la U-Net `unet_db3` como línea base, una sola vez | Se lanza `evaluar_test.py --abrir-test --modelos sam,unet` |
| 2026-11-05, 18:15 | La cola exploratoria se detuvo a las 17:13, durante E2a semilla 0, sin error en el log ni reinicio de Windows. Lo más probable es que se reiniciara la app que alojaba la terminal (no confirmado) | E1 quedó completa. Se relanza en una ventana independiente de la app, y ahora el preentrenamiento se guarda para no repetirlo. Se repite solo E2a semilla 0; el protocolo no cambia |
| 2026-11-05, 14:25 | **Desviación, decidida sin ver ningún resultado.** El primer intento se detuvo al preparar los chips, antes de cualquier predicción, porque Paraguay_34417 (test) no tiene ningún píxel con dato del radar pero sí 50 026 píxeles de agua etiquetada. Regla agregada: ese chip se queda con la máscara vacía para todos los métodos, así que todo su agua cuenta como FN. Se reporta además la métrica sin ese chip | Se relanza la evaluación |
| 2026-11-05, 14:30 | **Evaluación final en test y Bolivia**, una sola vez. Se descargaron 105 chips | Candidato SAM: test **0.642**, Bolivia **0.657**: cruza 0.60 en ambos. U-Net: test 0.666, Bolivia 0.628. La U-Net gana dentro de las regiones de entrenamiento y SAM, en la región no vista (sección 3h) |
| 2026-11-05, 18:14–18:23 | Tres relanzamientos de E2a semilla 0 (18:14, 18:20 y 18:23). La primera ventana se congeló por el modo de selección de la consola (QuickEdit), que se desactivó. La segunda se cerró sin causa identificada, y la tercera corrió oculta | Ningún relanzamiento produjo resultados |
| 2026-11-05, después de las 18:23 | **El equipo canceló la exploración restante.** Ya no quedan procesos de entrenamiento | E2a y E2b quedan sin resultado. E1, R1 y R2 sí terminaron. No se entrena nada más salvo que surjan cambios tras la reunión con el profesor |
| 2026-11-05, 21:45 | El equipo consiguió los PDF de DeepSARFlood y Wang et al., y se leyeron completos | Se corrigen las secciones 5, 5b y 5c, y se agrega la 5d. Ningún resultado nuestro cambia |
| 2026-11-06 | Póster y abstract de WAIMLAp rehechos con los resultados finales y las correcciones de los revisores, con asistencia de Claude Code (Opus 5.5). Figuras con `tools/figuras_poster_waimlap.py`: solo inferencia con los pesos guardados, y cada máscara reproduce el IoU por chip ya guardado | Archivos en `docs/waimlap_2026/`. No se entrenó nada ni se tomó ninguna decisión nueva sobre test |
| 2026-11-12 | Diagnóstico exploratorio SAM contra U-Net (`tools/diagnostico_sam_vs_unet.py`) y lectura de 7 trabajos más, con asistencia de Claude Code. Solo inferencia sobre train y valid con los modelos ya entrenados; test y Bolivia no se usaron | Los dos modelos encuentran el 97 % del agua oscura en VH y fallan en la que no se ve oscura (37 % del agua). Ninguno memoriza train. Detalle y propuesta de siguiente paso en `docs/siguiente_paso_sam_vs_unet.md`; nada de eso se ha ejecutado |
| 2026-11-12, 15:15 | Pre-registro de la validación cruzada por región (sección 2e) y lanzamiento (`tools/validacion_por_region.py`) | Pausada por el equipo a las 17:30 (3/10) y a las 22:19 (5/10); los pliegues guardados se reutilizan |
| 2026-11-13, 13:58 | Validación cruzada por región terminada (10/10, solo train + valid) | SAM 0.554 contra U-Net 0.571 en región no vista: −0.017, IC por región [−0.067, +0.029]. **Sin evidencia de diferencia** según la regla. SAM cae menos al cambiar de región (0.025 contra 0.055), pero no es concluyente (sección 3k) |

## 5. De dónde salen las ideas

| Trabajo | Qué hicieron | IoU de agua reportado |
|---|---|---|
| Garg et al. 2023, *Cross Modal Distillation for Flood Extent Mapping* (arXiv 2302.08180) | DeepLabv3+ ajustado completo, solo S1, 252 chips de entrenamiento; VV recortada a [−20, 0] dB y VH a [−30, 0] dB | 0.676 global en test (16 m); Otsu 0.546 |
| TerraMind tiny con S1 (github.com/Sina-Mohammadi/terramind-sentinel1-water-segmentation) | Ajuste completo, VV/VH normalizados con media y desviación fijas, rotaciones y espejados, 50 épocas | 0.553 en test |
| SAM / SAM 2 con LoRA (github.com/aksaN000/Sam2-sar-Flood-mapping) | LoRA de rango 8 en q y v de todos los bloques, sin prompts, BCE + Dice, coseno con calentamiento | En Pakistán 2022, fuera de dominio: con solo VV, 0.56–0.66; con VH en la entrada, 0.09–0.14 (según el README, no verificado) |
| Yadav et al. 2022, Frontiers in Remote Sensing | Attentive U-Net, VV+VH+ratio | ~0.67 |
| SAMed (arXiv 2304.13785) | LoRA en el codificador de SAM y entrenamiento sin prompts | Médico: base del diseño "sin prompts" |
| *Agregado el 05-nov, después de test:* Bonafilia et al. 2020 (CVPRW), el paper del dataset | FCNN ResNet50; IoU **medio por chip** | Test 0.31 (manuales) a 0.41 (débiles S2); Bolivia 0.35 a 0.39 |
| DeepSARFlood (Sharma y Saharia 2025, *Science of Remote Sensing* 11) | Ensambles de CNN y ViT con radar, terreno y agua permanente, y etiquetas débiles propias | 0.723 en el test oficial con el mejor modelo individual; 0.715 con el ensamble. No declara cómo agrega el IoU (ver 5d) |
| PANGAEA (arXiv 2412.04204) | Benchmark de modelos fundacionales geoespaciales | mIoU de 2 clases ~0.89–0.91 (otra métrica); una U-Net supervisada supera a la mayoría de los modelos fundacionales |
| Prithvi-CAFE (arXiv 2601.02315) | Prithvi con fusión adaptativa | ~0.83 en test; en Bolivia, TerraMind 0.62 contra U-Net 0.60 (entradas multimodales) |
| Wang, Neal, Atkinson y Zhang (preprint SSRN 5373217, sin revisión por pares) | Cinco métodos de ajuste eficiente de SAM ViT-B sobre UrbanSARFloods; LoRA el mejor | Con LoRA, Weihui 0.766 y Jubba 0.951, cifras que no cuadran con su F1 ni su recall. **Corrección:** el "~0.70 a ~0.90" que anoté antes no es una mejora, sino el nivel aproximado de esas dos regiones ya ajustadas (ver 5d) |
| Banerjee y Daou 2026 (arXiv 2606.16302) | U-Net contra SegFormer en SAR, solo agua de inundación | IoU 0.38–0.42 (clase distinta); al ajustar en Sen1Floods11, la diferencia entre arquitecturas cae dentro del ruido |

### 5b. Estado del arte desde 2020 (búsqueda del 05-nov, después de test)

Casi ningún número es comparable entre papers, porque cambian la métrica (IoU global, medio por chip o mIoU de 2 clases), la partición, la clase (toda el agua o solo la inundación) y la entrada (radar, óptico o ambos).

1. **Modelos fundacionales de observación de la Tierra** (Prithvi, TerraMind, CROMA, DOFA, SSL4EO-S12; lo dominante desde 2023).
   * Reportan un IoU de agua de ~0.80–0.83 en test.
   * Casi siempre usan Sentinel-2 o radar con óptico, no solo radar, así que no se comparan con lo nuestro.
   * En el benchmark PANGAEA, **una U-Net supervisada supera a la mayoría de los modelos fundacionales** en Sen1Floods11, igual que en nuestro trabajo.
   * En Bolivia, Prithvi-CAFE reporta TerraMind 0.62 contra U-Net 0.60. Es el mismo patrón que vimos: el preentrenado generaliza algo mejor a la región no vista.
2. **Solo radar (lo comparable con nosotros).**
   * DeepSARFlood (2025): 0.72 en test. **Corrección tras leerlo:** no es solo radar, porque también usa terreno y agua permanente, así que no es comparable con nosotros (ver 5d).
   * Una U-Net comunitaria con ResNet34 sobre VV y VH: 0.675 en el test oficial.
   * Banerjee y Daou (2026) comparan U-Net y SegFormer: al ajustar en Sen1Floods11, la diferencia cae dentro del ruido. Miden solo inundación, sin agua permanente.
3. **SAM en radar.**
   * Wang et al. (preprint en SSRN) prueban cinco métodos de ajuste eficiente sobre UrbanSARFloods, otro dataset, y LoRA es el mejor (ver 5d).
   * No se encontró un benchmark publicado de SAM sobre Sen1Floods11 con solo radar. Nuestro resultado (test 0.642, Bolivia 0.657) llena ese hueco.

**Lectura para nuestro trabajo:**
* Nuestros números son competitivos para solo radar y 250 chips.
* Que la U-Net le gane a SAM coincide con PANGAEA.
* El siguiente paso natural sería un modelo preentrenado con radar en vez de SAM, aunque ya solo como exploración, porque test se usó.

### 5c. Lectura de los trabajos comparables (05-nov, después de test)

Solo se leyeron trabajos con **Sentinel-1 solamente, sobre las etiquetas manuales de Sen1Floods11**. Los PDF están en `referencias/`, que se ignora en git. Las dos últimas filas se actualizaron después, al conseguir esos papers; ninguno de los dos cumple ese criterio (ver 5d).

| Trabajo | Qué es comparable | Números | Qué nos aporta |
|---|---|---|---|
| **Garg et al. 2023** (Google, arXiv 2302.08180). Leído completo | Solo VV y VH, partición oficial, **IoU global** (TP, FP y FN sumados sobre todas las imágenes), recorte fijo en dB (VV [−20, 0], VH [−30, 0]) | DeepLabv3+ con etiquetas manuales: **0.676 ± 0.005** (5 corridas, a 16 m). Solo etiquetas débiles de Sen1Floods11: 0.678 (**no mejora**). Débiles corregidas con el mapa de ocurrencia de agua JRC: 0.706. Destilación desde un maestro S1+S2: 0.719 (0.727 a 10 m) | (1) Nuestra U-Net (0.666) y SAM (0.642) están en su nivel con etiquetas manuales. (2) **Las etiquetas débiles solas no ayudan**: la 3c solo vale con corrección JRC o con destilación desde un maestro con Sentinel-2. (3) Tversky focal le sumó +0.019 en valid, en la misma dirección que nuestro +0.014. (4) Una salida de mayor resolución ayudó (+0.011), lo que apoya la hipótesis de que la máscara de 256×256 de SAM pesa. (5) Reportan 5 corridas: conviene pasar de 3 a 5 semillas |
| **sar-flood-unet** (repositorio comunitario; README y código leídos) | Solo VV y VH en dB, recorte por percentiles 0.5–99.5 de train, partición oficial, **IoU global**, D4 | U-Net con ResNet34 de ImageNet: **test 0.675, Bolivia 0.576**. **Umbral fijo en VH con filtro de mediana 5×5, ajustado en train: test 0.592, Bolivia 0.617** | (1) **Nuestro Otsu por chip (0.263 / 0.467) es una línea base débil.** Un umbral fijo ajustado en train con filtro de speckle es mucho más fuerte y en Bolivia supera a su U-Net. (2) En Bolivia, nuestro SAM (0.657) y nuestra U-Net (0.628) superan ese umbral filtrado (0.617), pero por poco. (3) Ellos excluyen Paraguay_34417; lo comparable con eso es nuestro "sin ese chip": SAM 0.653 y U-Net 0.677 |
| **Banerjee y Daou 2026** (arXiv 2606.16302). Leído | Solo radar y etiquetas manuales, pero en **3 clases** (inundación y agua permanente por separado, con JRC) y sobre 27 chips de test. **Solo parcialmente comparable** | Clase inundación: U-Net 0.408 contra SegFormer-b2 0.418. Diferencia +0.026 [−0.003, +0.057], **p = 0.34** | (1) En Sen1Floods11, la diferencia entre arquitecturas cae dentro del ruido entre chips, como entre nuestra U-Net y SAM. (2) Usan la misma estadística que nosotros (Wilcoxon y bootstrap por chip), lo que respalda nuestra metodología. (3) Preentrenan con ETCI (66 000 teselas de Sentinel-1), una fuente extra de datos de radar. (4) La TTA aporta poco, como nos pasó a nosotros |
| **DeepSARFlood** (Sharma y Saharia 2025). Leído completo el 05-nov | Solo la partición: test oficial de 90 chips. Usa radar con terreno y agua permanente, y 4 441 parches con etiquetas débiles propias. **No comparable** | 0.723 con Swin-V2 solo; 0.715 con el ensamble. Agregación del IoU no declarada. El 0.755 que anoté de otro resumen no aparece en esta versión | Información óptica durante el entrenamiento como posible vía de mejora; respalda no usar etiquetas débiles crudas (ver 5d) |
| **Wang, Neal, Atkinson y Zhang** (preprint SSRN 5373217). Leído completo el 05-nov | Solo el método: SAM ViT-B con LoRA. Usa UrbanSARFloods con una imagen previa al evento. **Cifras no comparables** | Con LoRA, Weihui 0.766 y Jubba 0.951. Estos IoU no cuadran con su F1 ni su recall | **Referencia de método, no de cifras:** ajustar el codificador es lo que importa, y LoRA gana entre cinco métodos (ver 5d) |

**Qué haríamos con esto** (todo exploratorio, porque test ya se usó):
1. **Agregar una línea base clásica fuerte**: umbral fijo en VH ajustado solo en train, con filtro de mediana 5×5. Es barato y hace más justa la comparación del E3. Se reportaría como línea base decidida después de ver test.
2. **Pasar a 5 semillas** las configuraciones finales (SAM y U-Net), como Garg.
3. **Si se hace la 3c**, no con etiquetas débiles crudas, sino corregidas con la ocurrencia de agua JRC (Garg: +0.03).
4. **Probar la capa JRC de agua permanente como canal extra** (DeepSARFlood, Banerjee). Sen1Floods11 la trae por chip.

Hallazgos en el código del repo que motivaron el paso 1: `baseline.normalize_band` (percentiles por chip), `config.BAND_USED = "VH"` y `decoder_ft._prompt_batch` (positivos dentro de Otsu incluso en chips sin agua real).

### 5d. DeepSARFlood y Wang et al., leídos completos (05-nov, 21:45)

El equipo consiguió ambos PDF, que no están en `referencias/`. Las comprobaciones de abajo son cálculo nuestro sobre sus tablas. Con una sola matriz de confusión, F1 = 2PR / (P + R) e IoU = 1 / (1/P + 1/R − 1), y el IoU nunca supera al F1 ni al recall. Que no cuadre no prueba un error de los autores: lo más probable es que agreguen el IoU de otra forma, y ninguno de los dos papers lo aclara.

**DeepSARFlood** (Sharma y Saharia 2025, [doi:10.1016/j.srs.2025.100203](https://doi.org/10.1016/j.srs.2025.100203)).

* **Por qué no se compara:**
  * Entrada: VV, VH, elevación, pendiente, HAND y agua permanente.
  * Entrenamiento: 4 441 parches con etiquetas débiles propias, 1 120 de Sen1Floods11 y 3 321 de otros eventos.
  * Validación: 341 chips manuales.
  * No reporta Bolivia. Sus otras pruebas (Pakistán y Assam) se miden contra etiquetas de Sentinel-2.
* **El 0.72 no es del ensamble:** Swin-V2 solo da 0.723 y el ensamble 0.715 en test. El ensamble sí gana en los dos casos de estudio.
* **Lo que aporta:**
  * El modelo base da 0.682 y sube a 0.712 al predecir también el índice óptico MNDWI durante el entrenamiento.
  * Garg et al. llegan a 0.719 con una idea parecida. **Hipótesis nuestra, sin probar:** lo que nos separa de 0.72 sería sobre todo información óptica al entrenar, no otra arquitectura.
  * Respalda no usar las etiquetas débiles crudas de Sen1Floods11: conserva 1 120 de 4 385 parches, con etiquetas rehechas.
  * Usa Tversky con más peso en los FN (α 0.7 en su fórmula), igual que nuestro β 0.7.

**Wang et al.** (preprint sin revisión por pares, [SSRN 5373217](https://ssrn.com/abstract=5373217)).

* **Por qué no se compara:**
  * Dataset: UrbanSARFloods, con 4 502 muestras de entrenamiento.
  * Entrada: VH antes del evento, y VH y VV después. La imagen previa le da información de cambio que nosotros no tenemos.
  * Evaluación: dos regiones, con una sola semilla.
* **Diferencia de diseño:** con LoRA congela el decodificador (366 mil parámetros entrenables). Nosotros entrenamos LoRA con el decodificador completo (4.35 millones). No declara el rango de LoRA.
* **Lo que aporta:**
  * LoRA obtiene el mejor IoU entre cinco métodos, y concluye que hace falta ajustar el codificador. Coincide con nuestro salto en valid de 0.472 a 0.60 al agregar LoRA, aunque eso no prueba que la causa sea la misma.
  * Explica el fallo de SAM original por su preferencia por objetos de bordes regulares, en línea con H1.
  * Propone a futuro combinar LoRA en el codificador con otro ajuste en el decodificador. Nosotros ya entrenamos LoRA con el decodificador completo, aunque no es exactamente lo mismo.
* **Dos cautelas sobre el texto:**
  * El resumen dice que el IoU sube "from 70% to 90%", pero SAM original parte de 0.70 %. Esos 70 % y 90 % son el nivel de las dos regiones ya ajustadas.
  * Una parte asocia el exceso de parámetros con el sobreajuste, y otra con el subajuste.

| Tabla | Modelo | Recall | F1 reportado | IoU reportado | IoU esperado con su P y R |
|---|---|---|---|---|---|
| DeepSARFlood, test (Tabla 3) | Swin-V2 | 86.69 | 78.91 | 72.26 | 70.72 |
| DeepSARFlood, Pakistán (Tabla 5) | Swin-V2 | 91.75 | 84.42 | 73.05 | 73.05 |
| Wang, Weihui (Tabla 1) | LoRA | 83.61 | 82.67 | 76.61 | 70.45 |
| Wang, Jubba (Tabla 2) | LoRA | 92.10 | 90.01 | **95.07** | 81.84 |
| Wang, Jubba (Tabla 2) | Prompt | 37.68 | 51.44 | **91.29** | 34.63 |

* **DeepSARFlood:** sus Tablas 5 y 6 cuadran exactas. En la Tabla 3, de donde sale el 0.72, el F1 reportado (78.91) tampoco coincide con el que dan su precisión y recall (82.85).
* **Wang et al.:** su F1 sí cuadra en todas las filas, pero el IoU supera al F1 y al recall en 8 de 12 filas. Promediar sobre dos clases no lo explica en 7 de esas 8.

**Efecto en nuestro trabajo:** ningún resultado cambia. Sigue en pie que no encontramos un resultado publicado de SAM sobre Sen1Floods11 con solo radar, y la búsqueda no fue exhaustiva.

## 6. Firmas del equipo sobre los pre-registros H4 y H5

| Integrante | Leí y acepto o corrijo las expectativas (indicar cuáles cambié) | Fecha |
|---|---|---|
| John Manuel Arzapalo Arana | | |
| Luis Alberto Miranda Mallqui | | |
| Juan Diego Zavala Viscardo | | |
