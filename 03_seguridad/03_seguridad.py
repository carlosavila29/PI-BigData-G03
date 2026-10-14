# Databricks notebook source
# MAGIC %md
# MAGIC # 03_seguridad · Clasificación, anonimización y control de acceso (A2.4)
# MAGIC **Grupo 03 · Retail / Comercio.** Marco: Ley 1581 de 2012 y Decreto 1377 de 2013.
# MAGIC
# MAGIC **Hallazgo previo.** El dataset del caso es agregado por tienda, departamento y semana: **no contiene datos personales**
# MAGIC (nombres, documentos, tarjetas de fidelización). La Ley 1581 protege datos de personas naturales, así que aquí no aplica
# MAGIC directamente. Sí hay información comercial confidencial de la cadena, y se trata con el mismo rigor como ejercicio y
# MAGIC porque un caso real (compras por cliente) sí traería datos personales.

# COMMAND ----------

from pyspark.sql import functions as F
ESQUEMA = "workspace.default"
v = spark.table(f"{ESQUEMA}.plata_ventas_enriquecida")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Clasificación de columnas por nivel de sensibilidad
# MAGIC Niveles: **Público**, **Interno**, **Confidencial** (comercial), **Personal** (Ley 1581), **Sensible** (art. 5 Ley 1581).

# COMMAND ----------

clasificacion = [
    ("Store",        "Interno",       "Identifica un establecimiento de la cadena; no identifica personas.", "Hash SHA-256 con sal"),
    ("Dept",         "Interno",       "Categoría comercial interna.", "Ninguna"),
    ("Date",         "Interno",       "Semana de venta; combinada con tienda permite inferir operación.", "Generalizar a mes"),
    ("Weekly_Sales", "Confidencial",  "Cifra de negocio de la cadena; ventaja competitiva.", "Acceso por rol; redondeo en vistas externas"),
    ("Size",         "Interno",       "Tamaño del local; con Store permite reidentificar la tienda.", "Generalizar a rangos"),
    ("Type",         "Público",       "Formato A/B/C, sin riesgo.", "Ninguna"),
    ("Temperature",  "Público",       "Dato climático de fuente abierta.", "Ninguna"),
]
display(spark.createDataFrame(clasificacion, ["columna", "nivel", "justificacion", "tratamiento"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Anonimización: antes y después
# MAGIC - **Técnica 1, hash SHA-256 con sal:** `Store` → `store_hash`.
# MAGIC - **Técnica 2, generalización / enmascaramiento:** `Date` → `mes`, `Size` → rango, `Store` → máscara `S-**`.
# MAGIC
# MAGIC La sal está en el código solo como ejemplo del ejercicio. En un caso real iría en un secreto (`dbutils.secrets`).

# COMMAND ----------

SAL = "pi_g03_demo"
anon = (v.select("Store", "Dept", "Date", "Size", "Weekly_Sales")
        .withColumn("store_hash", F.sha2(F.concat(F.col("Store").cast("string"), F.lit(SAL)), 256))
        .withColumn("store_mask", F.concat(F.lit("S-"), F.lit("**")))
        .withColumn("mes", F.date_format("Date", "yyyy-MM"))
        .withColumn("size_rango", F.when(F.col("Size") < 80000, "<80k").when(F.col("Size") < 150000, "80k-150k").otherwise(">=150k")))
print("ANTES")
display(anon.select("Store", "Date", "Size").limit(5))
print("DESPUÉS")
display(anon.select("store_hash", "store_mask", "mes", "size_rango").limit(5))

# COMMAND ----------

vista = anon.select("store_hash", "Dept", "mes", "size_rango", "Weekly_Sales")
vista.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.plata_ventas_anonimizadas")
assert "Store" not in vista.columns and "Date" not in vista.columns
print("plata_ventas_anonimizadas creada sin Store ni Date en claro")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Control de acceso por rol
# MAGIC Modelo propuesto sobre el esquema. En Databricks Free Edition la gestión de grupos puede estar limitada, por eso cada
# MAGIC sentencia se intenta y se reporta; la matriz sigue siendo el entregable de diseño.
# MAGIC
# MAGIC | Rol | `plata_ventas_enriquecida` (en claro) | `plata_ventas_anonimizadas` | `plata_cuarentena_*` |
# MAGIC |---|---|---|---|
# MAGIC | `ingeniero_datos` | SELECT, MODIFY | SELECT, MODIFY | SELECT, MODIFY |
# MAGIC | `analista_negocio` | SELECT | SELECT | sin acceso |
# MAGIC | `auditor_externo` | sin acceso | SELECT | sin acceso |

# COMMAND ----------

grants = [
    f"GRANT SELECT ON TABLE {ESQUEMA}.plata_ventas_enriquecida TO `analista_negocio`",
    f"GRANT SELECT ON TABLE {ESQUEMA}.plata_ventas_anonimizadas TO `analista_negocio`",
    f"GRANT SELECT ON TABLE {ESQUEMA}.plata_ventas_anonimizadas TO `auditor_externo`",
    f"GRANT SELECT, MODIFY ON SCHEMA {ESQUEMA} TO `ingeniero_datos`",
]
resultado = []
for g in grants:
    try:
        spark.sql(g); resultado.append((g, "aplicado"))
    except Exception as e:
        resultado.append((g, "no aplicado: " + str(e)[:120]))
display(spark.createDataFrame(resultado, ["sentencia", "estado"]))
