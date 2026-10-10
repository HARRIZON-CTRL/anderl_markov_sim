"""Cifras reportadas por Anderl et al. (2016), IJRM 33(3), y supuestos de la simulación.

Regla del repositorio:
    * Todo número que viene del artículo lleva la tabla de origen.
    * Todo lo que NO viene del artículo está en SUPUESTOS, con su justificación,
      y se declara en el README, en el notebook y en la sustentación.
    * Los dos parámetros que se calibraron (no se inventaron ni se tomaron del paper)
      están en CALIBRADOS, con la regla de calibración documentada en src/calibrate.py.

Todo el ejercicio usa el Data Set 1 (DS1, agencia de viajes): es la base con recorridos
más largos, lo más parecido al ciclo de decisión largo de un posgrado.
"""

# =============================================================================
# 1. DATOS REPORTADOS POR EL ARTÍCULO — DS1
# =============================================================================

# Tabla 1 — descripción de los recorridos
DS1 = {
    "industria": "Agencia de viajes",
    "n_canales": 8,
    "n_clics": 1_478_359,
    "n_recorridos": 600_978,
    "n_largo_ge2": 206_519,     # recorridos con 2 o más clics
    "n_largo_ge5": 48_344,      # recorridos con 5 o más clics
    "largo_media": 2.46,
    "largo_de": 8.860,
    "n_conversiones": 9_860,
    "tasa_conversion": 0.0164,
}

# Tabla 3 — canales de DS1 y participación de cada uno en el total de clics.
# (DS1 no registra entradas directas / Type-in.)
CANALES = ["SEA", "SEO", "PriceComp", "Display", "Newsletter", "Retargeting", "Affiliate", "Other"]
NOMBRES = {
    "SEA": "Búsqueda pagada", "SEO": "Búsqueda orgánica", "PriceComp": "Comparador de precios",
    "Display": "Display", "Newsletter": "Newsletter", "Retargeting": "Retargeting",
    "Affiliate": "Afiliado", "Other": "Otro",
}
PARTICIPACION_CLICS = {
    "SEA": 0.234, "SEO": 0.068, "PriceComp": 0.037, "Display": 0.595,
    "Newsletter": 0.012, "Retargeting": 0.004, "Affiliate": 0.025, "Other": 0.026,
}

# Tabla 5 — AUC y lift del decil superior, DS1, validación cruzada de 10 particiones.
#                   AUC dentro, AUC fuera, lift dentro, lift fuera
RESULTADOS_PAPER_DS1 = {
    "Markov orden 1": (0.7398, 0.7396, 2.9056, 2.9075),
    "Markov orden 2": (0.8214, 0.8207, 4.7181, 4.6937),
    "Markov orden 3": (0.8323, 0.8305, 5.2798, 5.2393),
    "Markov orden 4": (0.8357, 0.8294, 5.3587, 5.2913),
    "Último clic":    (0.7398, 0.7406, 2.9056, 2.9175),
    "Primer clic":    (0.7133, 0.7136, 2.4333, 2.4296),
    "Logit 1":        (0.8185, 0.8185, 4.9226, 4.9260),
    "Logit 2":        (0.8278, 0.8274, 5.1854, 5.1856),
}

# Tabla 7 — atribución en % para DS1: Markov de orden 3, último clic y primer clic.
# Las columnas de último y primer clic son la proporción de conversiones cuyo
# último (o primer) clic fue ese canal: un dato descriptivo que el simulador debe reproducir.
TABLA7_DS1 = {
    #               Markov 3, último clic, primer clic
    "SEA":         (46.29, 53.19, 56.36),
    "SEO":         (19.54, 16.76, 16.67),
    "PriceComp":   (5.29, 4.78, 6.05),
    "Display":     (0.93, 0.14, 0.21),
    "Newsletter":  (4.47, 2.93, 4.28),
    "Retargeting": (1.25, 0.67, 0.78),
    "Affiliate":   (19.71, 20.17, 13.66),
    "Other":       (2.52, 1.36, 1.97),
}

# Tabla 8 — coeficientes del Logit 1 en DS1 (logit de la conversión sobre el NÚMERO
# de clics de cada canal). Se usan como base del mecanismo de conversión simulado.
TABLA8_DS1_BETA = {
    "SEA": 0.443, "SEO": 0.388, "PriceComp": 0.002, "Display": -3.427,
    "Newsletter": 0.616, "Retargeting": 0.310, "Affiliate": 1.152, "Other": 0.032,
}
TABLA8_DS1_INTERCEPTO = -4.505

# =============================================================================
# 2. SUPUESTOS (no reportados en el artículo)
# =============================================================================
SUPUESTOS = {
    # S1. Transición entre canales. Con probabilidad `arrastre` el siguiente clic repite
    #     el canal anterior (carryover, sección 6.2); si no, el canal se sortea según la
    #     participación de la Tabla 3. Con esta forma, la proporción de clics por canal
    #     converge a la Tabla 3 (se verifica en el notebook).
    "arrastre": 0.35,

    # S2. Largo del recorrido. P(L=1), P(2<=L<=4) y P(L>=5) salen de los conteos de la
    #     Tabla 1. Lo que no se reporta es la forma dentro de cada tramo: se reparte
    #     L=2,3,4 en 55/28/17 % y la cola L>=5 es 5 + binomial negativa, con media y
    #     dispersión elegidas para acercarse a la media (2,46) y la DE (8,86) de la Tabla 1.
    "p_largo_2_4": [0.55, 0.28, 0.17],
    "cola_media_extra": 8.86,
    "cola_dispersion": 0.096,
    "largo_max": 500,

    # S3. Mecanismo de conversión (el supuesto central). La probabilidad de comprar de
    #     un recorrido es un logit con tres partes:
    #       (a) conteos por canal con los coeficientes de la Tabla 8  -> REPORTADO (signos y
    #                                                                    proporciones); escala CALIBRADA
    #       (b) un peso extra al último canal (recencia)              -> CALIBRADO
    #       (c) efectos de secuencia (parejas y tríos ordenados al
    #           final del recorrido)                                   -> estructura SUPUESTA,
    #                                                                    magnitud CALIBRADA
    #       (d) efecto de volver: log(número de clics)                 -> CALIBRADO
    #     El intercepto se ajusta en cada simulación para que la tasa de conversión sea
    #     exactamente la de la Tabla 1 (1,64 %).
    #
    #     Estructura de (c), tomada de lo que el paper describe en las secciones 6 y 7:
    "efectos_pares": {   # (penúltimo canal, último canal): efecto base
        ("Display", "SEA"): 1.0,      # los banners abaratan la búsqueda posterior (sec. 6.1)
        ("Display", "SEO"): 1.0,      # idem
        ("SEA", "SEO"): 0.8,          # la búsqueda pagada derrama en la orgánica (sec. 6.2)
        ("Newsletter", "SEA"): 0.8,   # el newsletter motiva visitas por otros canales (sec. 6.1)
        ("Newsletter", "SEO"): 0.8,   # idem
        ("SEA", "SEA"): 0.5,          # carryover en búsqueda pagada (Tabla 9)
        ("SEO", "SEO"): 0.5,          # carryover en búsqueda orgánica (Tabla 9)
        ("Affiliate", "Affiliate"): 0.5,  # carryover en afiliados (Tabla 9)
    },
    "efectos_trios": {   # (antepenúltimo, penúltimo, último): efecto base
        ("Display", "SEO", "SEA"): 1.0,      # banner -> orgánica -> pagada: búsqueda madura
        ("Display", "Display", "SEA"): 0.8,  # exposición repetida a banners y luego búsqueda
        ("SEA", "SEO", "SEA"): 0.8,          # ida y vuelta entre búsquedas
    },

    # S4. Tiempo. La simulación no genera fechas, solo el orden de los clics; igual que el
    #     modelo del paper, que usa el orden y no el tiempo entre contactos.
}

# =============================================================================
# 3. PARÁMETROS CALIBRADOS (ver src/calibrate.py y results/calibracion.csv)
# =============================================================================
# Son 4 parámetros. Regla: se eligieron los valores que minimizan la distancia entre el AUC fuera de
# muestra simulado y el reportado en la Tabla 5 para cinco modelos ancla (último clic,
# primer clic, Logit 1, Markov de orden 2 y Markov de orden 3), sobre una muestra de
# calibración con una semilla distinta de la usada en el análisis final.
CALIBRADOS = {
    "escala_conteo": 0.45,  # (a) multiplica los coeficientes de la Tabla 8
    "peso_ultimo": 0.0,     # (b) recencia: la calibración la dejó en 0, porque los efectos
                            #     de secuencia al final del recorrido ya dan esa ventaja
    "escala_orden": 1.5,    # (c) multiplica los efectos de parejas y tríos de S3
    "efecto_retorno": 1.5,  # (d) efecto de log(número de clics): volver más de una vez
}
# Resultado de la calibración (muestra de calibración, AUC fuera de muestra):
#   último clic 0,767 (paper 0,741) · primer clic 0,716 (0,714) · Logit 1 0,832 (0,819)
#   Markov 2 0,830 (0,821) · Markov 3 0,821 (0,831) · pérdida 0,0010

SEMILLA = 2016              # análisis final
SEMILLA_CALIBRACION = 7     # calibración (distinta, para no calibrar y evaluar sobre lo mismo)
