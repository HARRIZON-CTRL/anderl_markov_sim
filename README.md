# Recreación de Anderl et al. (2016) con datos simulados

**Actividad final — Machine Learning II**

**Artículo:** Anderl, E., Becker, I., von Wangenheim, F. y Schumann, J. H. (2016). Mapping the customer journey: Lessons learned from graph-based online attribution modeling. *International Journal of Research in Marketing*, 33(3), 457–474.

**Relación con el trabajo de grado:** el trabajo de grado modela el recorrido de los prospectos de una escuela de negocios como un grafo (nodo = interacción) para estimar si y cuándo se matriculan. Este artículo es el antecedente directo: representa los recorridos de compra como grafos de Markov y evalúa si recordar el orden del recorrido mejora la predicción de la compra.

## Resultado en una línea

La simulación reproduce los datos descriptivos del paper y sus hallazgos predictivos: el orden 1 es igual al último clic, dar memoria produce un gran salto y después del orden 2 la ganancia se estanca. No reproduce la atribución por canal, porque el paper no publica la relación entre tipo de persona y canal. A la escala de la escuela (15.000–20.000 personas), el orden 2 basta y el orden 4 sobreajusta; con más acciones por persona (≥ 3), el orden 3 vuelve a ganar.

## Estructura

```
anderl_markov_sim/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_recreacion_anderl.ipynb   análisis completo, pasos 1 a 8 de la guía (ya ejecutado)
├── src/
│   ├── params.py      cifras del paper (Tablas 1, 3, 5, 7 y 8), supuestos y parámetros calibrados
│   ├── simulate.py    simulador de recorridos (4 pasos, comentado) y explicar_simulacion() para verlos uno a uno
│   ├── calibrate.py   regla de calibración del mecanismo de compra
│   ├── markov.py      grafos de Markov de orden k, predicción y efecto de remoción
│   ├── grafos.py      dibujos del grafo: orden 1 y matriz W, detalle de orden 2, remoción, esquema de la escuela
│   ├── baselines.py   último clic, primer clic, Logit 1 y Logit 2
│   └── evaluate.py    validación cruzada, AUC, lift, comparación con las Tablas 7 y 8, extensión
├── data/              (los datos se generan al correr el notebook; no se guardan por tamaño)
├── results/           calibración, validación cruzada, Tabla 5 comparada, extensión
└── figures/           grafos (orden 1, detalle de orden 2, remoción, escuela), validación, AUC paper vs. simulación, extensión
```

## Cómo ejecutar

En Anaconda Prompt, desde la carpeta del repositorio:

```bat
conda env create -f environment.yml
conda activate anderl
```

Luego, en VS Code (con las extensiones Python y Jupyter de Microsoft): abrir `notebooks/01_recreacion_anderl.ipynb`, elegir el kernel **anderl** (Select Kernel → Python Environments → anderl) y **Run All**.

El ambiente trae solo lo necesario: las librerías del análisis e `ipykernel`, que es lo que VS Code usa para correr el notebook. No hace falta instalar Jupyter.

**Paralelo.** Las 10 particiones de la validación cruzada son independientes y se reparten entre los núcleos del equipo con `joblib` (`N_JOBS = -1` en la primera celda). Además, las tablas de los Logit se arman con numpy en lugar de `pd.crosstab`. Con eso, el notebook completo pasó de unos 29 minutos a unos 2 minutos (medido con 2 núcleos; con más núcleos, menos). Los resultados son idénticos a los de la versión en serie.

Para pruebas, `MODO_RAPIDO = True` usa 2 particiones.

Si se cambia algo del ambiente, recrearlo desde cero:

```bat
conda deactivate
conda env remove -n anderl
conda env create -f environment.yml
```

La calibración no se corre desde el notebook: sus resultados están en `results/calibracion*.csv`. Para repetirla:

```python
from src.calibrate import calibrar, REJILLA, REJILLA_FINA
tabla, objetivo = calibrar(REJILLA)        # etapa gruesa
tabla_f, _ = calibrar(REJILLA_FINA)        # etapa fina
```

## Qué se toma del paper y qué se supone

| Elemento | Fuente |
|---|---|
| Recorridos, clics, % de largo ≥ 2 y ≥ 5, media y DE del largo, conversiones | Paper, Tabla 1 (DS1) |
| Participación de clics por canal | Paper, Tabla 3 |
| Efecto de cada clic por canal (signos y proporciones) | Paper, Tabla 8 (Logit 1) |
| Modelos, métricas y protocolo (10 particiones, AUC, lift) | Paper, sección 5 y Tabla 5 |
| **S1** Transición entre canales: 35 % repite el anterior; si no, sorteo según la Tabla 3 | Supuesto |
| **S2** Forma del largo dentro de cada tramo | Supuesto (reproduce media y DE) |
| **S3** Mecanismo de compra: conteos (Tabla 8) + recencia + secuencias finales + volver | Estructura supuesta; 4 magnitudes calibradas |
| **S4** Sin fechas: solo el orden de los clics | Supuesto (el modelo tampoco usa el tiempo) |
| Predicción del Markov: P(compra ∣ último estado), con suavizado hacia el orden inferior | Interpretación + decisión de implementación |
| Efecto de remoción por canal: se quitan a la vez todos los estados que terminan en el canal | Simplificación (el paper promedia por estado) |

## Regla de calibración

Las cuatro magnitudes de S3 se eligieron para que el AUC fuera de muestra de cinco modelos ancla (último clic, primer clic, Logit 1, Markov 2 y Markov 3) se pareciera al de la Tabla 5. Se hizo sobre una muestra de calibración con semilla distinta y en dos etapas de rejilla (108 + 72 combinaciones). Los anclas no cuentan como evidencia. La verificación real está en lo que no se usó: Markov 1 y 4, Logit 2, los lift y las Tablas 7 y 8.

| Parámetro | Valor | Significado |
|---|---|---|
| `escala_conteo` | 0,45 | Peso de los coeficientes de la Tabla 8 |
| `peso_ultimo` | 0,0 | Recencia extra (la calibración no la necesitó) |
| `escala_orden` | 1,5 | Peso de las secuencias al final del recorrido |
| `efecto_retorno` | 1,5 | Efecto de log(número de clics) |

## El grafo

La sección 5.1 del notebook dibuja el grafo de Markov construido con los datos simulados antes de usarlo (solo `grafo_escuela.png` es un esquema sin datos):

| Figura | Qué muestra |
|---|---|
| `figures/grafo_orden1.png` | Grafo de orden 1 (nodo = canal, con INICIO, CONVERSIÓN y NULO) y su matriz de transición W |
| `figures/grafo_recorridos.png` | Dos recorridos de la base (uno con compra, uno sin compra) dibujados como caminos sobre el grafo |
| `figures/grafo_orden2_detalle.png` | Cómo el nodo búsqueda pagada se parte en parejas en el orden 2: de 1,1 % a 4,1 % de conversión según el canal anterior (por eso la memoria sube el AUC) |
| `figures/grafo_orden2.png` | Grafo de orden 2 completo con los datos: 72 parejas en rejilla (columna = canal actual, fila = anterior) |
| `figures/grafo_por_orden.png` | El grafo de los datos en los órdenes 1 a 4: 8, 72, 540 y 2.926 nodos; en el orden 4, el 78 % tiene menos de 50 visitas |
| `figures/grafo_remocion.png` | Efecto de remoción en el grafo: sin búsqueda pagada, P(INICIO → CONVERSIÓN) cae de 1,64 % a 0,60 % |
| `figures/grafo_escuela.png` | Esquema ilustrativo (sin datos) del grafo para la escuela: experiencias transitorias, matrícula y NULO absorbentes |

## Resultados principales (DS1, AUC fuera de muestra, 10 particiones)

| Modelo | Paper | Simulación |
|---|---|---|
| Markov orden 1 | 0,740 | 0,758 |
| Markov orden 2 | 0,821 | 0,834 |
| Markov orden 3 | 0,831 | 0,832 |
| Markov orden 4 | 0,829 | 0,832 |
| Último clic | 0,741 | 0,758 |
| Primer clic | 0,714 | 0,718 |
| Logit 1 | 0,819 | 0,832 |
| Logit 2 | 0,827 | 0,851 |

**Extensión (cambia solo el número de personas):** un recorrido es una persona, no una acción. Con la base completa (600.978 personas), los órdenes 2, 3 y 4 empatan. A la escala de la escuela (15.000–20.000 personas, ~330 compras), el orden 2 es el mejor Markov, el orden 4 cae (0,792 a 20.000) y el Logit 2 se mantiene en ≈0,85; la DE entre particiones sube a ±0,05, del mismo tamaño que las diferencias.

**Análisis adicional (7.2):** 20.000 personas con al menos 3 acciones cada una (77.307 acciones). Todos los modelos mejoran y el orden 3 vuelve a ganar (0,853 frente a 0,831 del orden 2): cuánta historia tiene cada persona importa tanto como cuántas personas hay.
