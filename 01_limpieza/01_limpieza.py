# Databricks notebook source
# MAGIC %md
# MAGIC # 01_limpieza · Capa PLATA (A2.1)
# MAGIC **Grupo 03 · Retail / Comercio.** Parte de las tablas bronce de `00_ingesta`. Aplica 6 transformaciones documentadas y
# MAGIC registra las filas antes y después de cada una en `plata_log_transformaciones`.
# MAGIC
# MAGIC | # | Transformación | Tabla |
# MAGIC |---|---|---|
# MAGIC | T1 | Deduplicación por llave natural | train, features |
# MAGIC | T2 | Tipificación (fecha, numéricos, booleano; `NA` → nulo) | train, features, stores |
# MAGIC | T3 | Estandarización de categorías (`Type`) | stores |
# MAGIC | T4 | Tratamiento de nulos (MarkDown como ausencia real; CPI/Unemployment por arrastre) | features |
# MAGIC | T5 | Filtrado de atípicos (ventas negativas a cuarentena) | train |
# MAGIC | T6 | Unión de fuentes (ventas + indicadores + tiendas) | las tres |

# COMMAND ----------

from pyspark.sql import functions as F, Window

ESQUEMA = "workspace.default"
log = []   # (transformacion, tabla, filas_antes, filas_despues, nota)

def registrar(t, tabla, antes, despues, nota=""):
    log.append((t, tabla, int(antes), int(despues), nota))
    print(f"{t} · {tabla}: {antes:,} -> {despues:,}  {nota}")

train = spark.table(f"{ESQUEMA}.bronce_train")
feat = spark.table(f"{ESQUEMA}.bronce_features")
stores = spark.table(f"{ESQUEMA}.bronce_stores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 · Deduplicación
# MAGIC Llave natural de ventas: `(Store, Dept, Date)`; de indicadores: `(Store, Date)`. En la exploración local no había duplicados;
# MAGIC se deja el paso ejecutado para que quede comprobado en el propio workspace.

# COMMAND ----------

n0 = train.count(); train = train.dropDuplicates(["Store", "Dept", "Date"]); registrar("T1 deduplicación", "train", n0, train.count(), "llave Store+Dept+Date")
n0 = feat.count();  feat = feat.dropDuplicates(["Store", "Date"]);          registrar("T1 deduplicación", "features", n0, feat.count(), "llave Store+Date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 · Tipificación
# MAGIC Bronce guarda todo como texto. Aquí se convierte a su tipo real y el literal `NA` pasa a nulo verdadero.
# MAGIC Se registran los nulos antes y después para comprobar que no se perdió ni se inventó información.

# COMMAND ----------

def na_a_nulo(c):
    return F.when(F.trim(F.col(c)) == "NA", None).otherwise(F.trim(F.col(c)))

def bool_(c):
    return F.when(F.upper(F.trim(F.col(c))) == "TRUE", True).when(F.upper(F.trim(F.col(c))) == "FALSE", False)

train = train.select(
    F.col("Store").cast("int").alias("Store"), F.col("Dept").cast("int").alias("Dept"),
    F.to_date("Date", "yyyy-MM-dd").alias("Date"),
    F.col("Weekly_Sales").cast("double").alias("Weekly_Sales"), bool_("IsHoliday").alias("IsHoliday"))

num_feat = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]
feat = feat.select(F.col("Store").cast("int").alias("Store"), F.to_date("Date", "yyyy-MM-dd").alias("Date"),
                   *[na_a_nulo(c).cast("double").alias(c) for c in num_feat], bool_("IsHoliday").alias("IsHoliday"))

stores = stores.select(F.col("Store").cast("int").alias("Store"), F.trim("Type").alias("Type"), F.col("Size").cast("int").alias("Size"))

nulos_train = train.filter(F.col("Date").isNull() | F.col("Weekly_Sales").isNull()).count()
registrar("T2 tipificación", "train", train.count(), train.count(), f"filas con fecha o venta no convertible: {nulos_train}")
registrar("T2 tipificación", "features", feat.count(), feat.count(), "NA -> null en 9 columnas numéricas")
registrar("T2 tipificación", "stores", stores.count(), stores.count(), "Size a int")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 · Estandarización de categorías
# MAGIC `Type` debe ser A, B o C. Se pasa a mayúscula y se verifica que no queden valores fuera del dominio.

# COMMAND ----------

stores = stores.withColumn("Type", F.upper(F.col("Type")))
fuera = stores.filter(~F.col("Type").isin("A", "B", "C")).count()
registrar("T3 categorías", "stores", stores.count(), stores.count(), f"valores de Type fuera de A/B/C: {fuera}")
display(stores.groupBy("Type").agg(F.count("*").alias("tiendas"), F.min("Size").alias("size_min"), F.max("Size").alias("size_max")).orderBy("Type"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 · Tratamiento de nulos
# MAGIC - **MarkDown1-5:** el nulo no es un error, es que la promoción no existía (el programa de descuentos arranca en nov. 2011).
# MAGIC   No se imputa con un promedio: se crea `hay_promocion` y los MarkDown nulos pasan a 0.
# MAGIC - **CPI y Unemployment:** los nulos son las últimas semanas de cada tienda; se arrastra el último valor conocido por tienda.
# MAGIC   Esas semanas caen fuera del periodo de ventas (train), así que no afectan el análisis; se documenta igual.

# COMMAND ----------

mk = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
nulos_mk_antes = feat.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in mk]).first().asDict()
feat = feat.withColumn("hay_promocion", F.greatest(*[F.col(c).isNotNull() for c in mk]))
for c in mk:
    feat = feat.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

w = Window.partitionBy("Store").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)
for c in ["CPI", "Unemployment"]:
    feat = feat.withColumn(c, F.last(c, ignorenulls=True).over(w))

quedan = feat.filter(F.col("CPI").isNull() | F.col("Unemployment").isNull()).count()
registrar("T4 nulos", "features", feat.count(), feat.count(), f"MarkDown nulos antes={nulos_mk_antes}; CPI/Unemployment aún nulos={quedan}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 · Filtrado de atípicos
# MAGIC `Weekly_Sales < 0` son devoluciones que superan las ventas de la semana. No se borran: pasan a una tabla de cuarentena
# MAGIC (`plata_cuarentena_ventas_negativas`) para que el área de inventario las pueda revisar, y salen de la tabla de análisis.

# COMMAND ----------

n0 = train.count()
negativas = train.filter(F.col("Weekly_Sales") < 0)
negativas.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.plata_cuarentena_ventas_negativas")
train = train.filter(F.col("Weekly_Sales") >= 0)
registrar("T5 atípicos", "train", n0, train.count(), f"enviadas a cuarentena: {n0 - train.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 · Unión de fuentes
# MAGIC `ventas` LEFT JOIN `features` por `(Store, Date)` y LEFT JOIN `stores` por `Store`. Se verifica que el número de filas
# MAGIC no cambie (una unión mal hecha multiplica filas) y se cuentan las ventas sin indicadores.

# COMMAND ----------

n0 = train.count()
ventas = (train.alias("v")
          .join(feat.drop("IsHoliday").alias("f"), ["Store", "Date"], "left")
          .join(stores.alias("s"), "Store", "left"))
n1 = ventas.count()
sin_ind = ventas.filter(F.col("Temperature").isNull()).count()
sin_tienda = ventas.filter(F.col("Type").isNull()).count()
registrar("T6 unión", "plata_ventas", n0, n1, f"sin indicadores={sin_ind}; sin tienda={sin_tienda}")
assert n0 == n1, "La unión cambió el número de filas"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persistencia de la capa plata y bitácora

# COMMAND ----------

for nombre, df in {"plata_ventas": train, "plata_features": feat, "plata_stores": stores, "plata_ventas_enriquecida": ventas}.items():
    df.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.{nombre}")

bit = spark.createDataFrame(log, ["transformacion", "tabla", "filas_antes", "filas_despues", "nota"])
bit.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{ESQUEMA}.plata_log_transformaciones")
display(bit)
