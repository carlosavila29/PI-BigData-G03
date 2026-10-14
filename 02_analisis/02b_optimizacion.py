# Databricks notebook source
# MAGIC %md
# MAGIC # 02b_optimizacion · Tiempos antes y después (A2.6)
# MAGIC **Grupo 03.** Tres técnicas, cada una medida 3 veces (se reporta la mediana).
# MAGIC 1. **Broadcast join** frente a sort-merge (hints de Spark SQL).
# MAGIC 2. **Particionamiento** por año frente a tabla sin particionar, en una consulta filtrada por fecha.
# MAGIC 3. **OPTIMIZE + Z-ORDER** por `Store`, en consulta filtrada por tienda.
# MAGIC
# MAGIC El caché (`.cache()`) no se usa: en cómputo serverless de Free Edition no está soportado.

# COMMAND ----------

import time, statistics
from pyspark.sql import functions as F
ESQUEMA = "workspace.default"
resultados = []

def medir(tecnica, variante, fn, n=3):
    tiempos = []
    for _ in range(n):
        t0 = time.time(); fn(); tiempos.append(time.time() - t0)
    resultados.append((tecnica, variante, round(statistics.median(tiempos), 2), round(min(tiempos), 2), round(max(tiempos), 2)))
    print(tecnica, variante, "mediana", round(statistics.median(tiempos), 2), "s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Técnica 1 · Broadcast join

# COMMAND ----------

def consulta_join(hint):
    return lambda: spark.sql(f"""
      SELECT /*+ {hint}(s) */ s.Type, SUM(v.Weekly_Sales) AS venta
      FROM {ESQUEMA}.plata_ventas v JOIN {ESQUEMA}.plata_stores s ON v.Store = s.Store
      GROUP BY s.Type""").collect()

medir("1. join", "antes: sort-merge", consulta_join("MERGE"))
medir("1. join", "después: broadcast", consulta_join("BROADCAST"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Técnica 2 · Particionamiento por año

# COMMAND ----------

base = spark.table(f"{ESQUEMA}.plata_ventas").withColumn("anio", F.year("Date"))
base.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.opt_ventas_plana")
base.write.format("delta").mode("overwrite").option("overwriteSchema", True).partitionBy("anio").saveAsTable(f"{ESQUEMA}.opt_ventas_part")

def consulta_anio(tabla):
    return lambda: spark.sql(f"SELECT Store, SUM(Weekly_Sales) FROM {ESQUEMA}.{tabla} WHERE anio = 2012 GROUP BY Store").collect()

medir("2. partición", "antes: sin particionar", consulta_anio("opt_ventas_plana"))
medir("2. partición", "después: partitionBy(anio)", consulta_anio("opt_ventas_part"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Técnica 3 · OPTIMIZE + Z-ORDER

# COMMAND ----------

def consulta_tienda():
    return spark.sql(f"SELECT Dept, SUM(Weekly_Sales) FROM {ESQUEMA}.opt_ventas_plana WHERE Store = 20 GROUP BY Dept").collect()

medir("3. Z-ORDER", "antes", consulta_tienda)
spark.sql(f"OPTIMIZE {ESQUEMA}.opt_ventas_plana ZORDER BY (Store)")
medir("3. Z-ORDER", "después", consulta_tienda)

# COMMAND ----------

tabla = spark.createDataFrame(resultados, ["tecnica", "variante", "mediana_s", "min_s", "max_s"])
display(tabla)
tabla.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.opt_resultados_tiempos")

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura honesta.** Con ~420 mil filas, las diferencias pueden ser pequeñas o incluso favorecer la variante "antes":
# MAGIC el costo fijo de Spark domina. Reportar lo medido y explicarlo; no forzar una mejora que los números no muestran.
