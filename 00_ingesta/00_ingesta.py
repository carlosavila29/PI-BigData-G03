# Databricks notebook source
# MAGIC %md
# MAGIC # 00_ingesta · Capa BRONCE
# MAGIC **Proyecto Integrador Big Data · Grupo 03 · Retail / Comercio (Walmart Store Sales Forecasting)**
# MAGIC
# MAGIC Lee los 4 CSV oficiales y los persiste *tal cual* como tablas Delta de la capa bronce (todo como texto, sin corregir nada).
# MAGIC La tipificación y la limpieza son la capa plata (`01_limpieza`).
# MAGIC
# MAGIC - **Fuente oficial:** Kaggle, Walmart Recruiting - Store Sales Forecasting, https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
# MAGIC - **Fecha de descarga:** 06/09/2026. Kaggle exige iniciar sesión y aceptar las reglas de la competencia; los 4 archivos se obtuvieron de un espejo público en GitHub (`vicky60629/Walmart-Store-Sales-Forecasting`) que se asume con el mismo contenido que la fuente oficial (no se pudo contrastar directamente).
# MAGIC - **Destino:** Unity Catalog, `workspace.default`

# COMMAND ----------

RUTA = "/Volumes/workspace/default/proyecto_integrador_grupo03"
ESQUEMA = "workspace.default"
FUENTE_URL = "https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data"
ARCHIVOS = {"train": "train.csv.gz", "test": "test.csv", "features": "features.csv", "stores": "stores.csv"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lectura y persistencia en bronce
# MAGIC Se lee con `inferSchema=False`: todo llega como `string`. Así el nulo `NA` de Kaggle sigue visible en bronce y la
# MAGIC decisión de qué hacer con él queda documentada en plata, no escondida en la lectura.

# COMMAND ----------

from pyspark.sql import functions as F

bronce = {}
for nombre, archivo in ARCHIVOS.items():
    df = (spark.read.option("header", True).option("inferSchema", False)
          .csv(f"{RUTA}/{archivo}")
          .withColumn("_fuente_url", F.lit(FUENTE_URL))
          .withColumn("_archivo", F.lit(archivo))
          .withColumn("_ingesta_ts", F.current_timestamp()))
    df.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.bronce_{nombre}")
    bronce[nombre] = spark.table(f"{ESQUEMA}.bronce_{nombre}")
    print(f"bronce_{nombre}: {bronce[nombre].count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validación 1: conteo de filas contra el archivo fuente
# MAGIC Los conteos esperados salen de la exploración local de la Fase 1.

# COMMAND ----------

ESPERADO = {"train": 421570, "test": 115064, "features": 8190, "stores": 45}
filas = [(n, bronce[n].count(), ESPERADO[n], bronce[n].count() == ESPERADO[n]) for n in ESPERADO]
display(spark.createDataFrame(filas, ["tabla", "filas_bronce", "filas_esperadas", "coincide"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validación 2: esquema

# COMMAND ----------

for n, df in bronce.items():
    print(f"--- bronce_{n}")
    df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validación 3: nulos por columna (incluye el literal `NA`)

# COMMAND ----------

def nulos_por_columna(df, nombre):
    cols = [c for c in df.columns if not c.startswith("_")]
    agg = [F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == "NA") | (F.trim(F.col(c)) == ""), 1).otherwise(0)).alias(c) for c in cols]
    fila = df.agg(*agg).first().asDict()
    total = df.count()
    return [(nombre, c, int(v), round(100.0 * v / total, 2)) for c, v in fila.items()]

res = []
for n, df in bronce.items():
    res += nulos_por_columna(df, f"bronce_{n}")
display(spark.createDataFrame(res, ["tabla", "columna", "nulos", "pct_nulos"]))
