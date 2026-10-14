# Databricks notebook source
# MAGIC %md
# MAGIC # 02_analisis · Capa PLATA → análisis (A2.2)
# MAGIC **Grupo 03 · Retail / Comercio.** Responde las 3 preguntas de negocio con 5 consultas (Spark SQL y DataFrame API).
# MAGIC Incluye 1 agregación con función de ventana (Q2) y 1 join entre 2 fuentes (Q3).
# MAGIC
# MAGIC | Pregunta | Consultas |
# MAGIC |---|---|
# MAGIC | P1 ¿Cuánto suben las ventas semanales en semanas festivas, por tipo de tienda? | Q1, Q2 |
# MAGIC | P2 ¿Dónde (tiendas, departamentos) se concentra la venta que hay que planear? | Q3 |
# MAGIC | P3 ¿Por qué varían las ventas: clima, combustible, economía o promociones? | Q4, Q5 |
# MAGIC
# MAGIC > Las 3 preguntas son la propuesta del equipo derivada de la pregunta general de la Fase 1. Ajustarlas si el docente las pide distintas.

# COMMAND ----------

from pyspark.sql import functions as F, Window
ESQUEMA = "workspace.default"
ventas = spark.table(f"{ESQUEMA}.plata_ventas_enriquecida")
ventas.createOrReplaceTempView("v")
spark.table(f"{ESQUEMA}.plata_stores").createOrReplaceTempView("t")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q1 · Venta promedio por semana festiva y tipo de tienda (P1)

# COMMAND ----------

q1 = spark.sql("""
SELECT Type AS tipo_tienda, IsHoliday AS festivo,
       COUNT(*) AS registros,
       ROUND(AVG(Weekly_Sales), 2) AS venta_promedio,
       ROUND(PERCENTILE_APPROX(Weekly_Sales, 0.5), 2) AS venta_mediana
FROM v GROUP BY Type, IsHoliday ORDER BY Type, IsHoliday
""")
display(q1)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación Q1.** En semanas festivas la venta promedio sube ≈6,4 % en tiendas A (20.055 → 21.333) y ≈10,0 % en B (12.203 → 13.420); en C prácticamente no cambia (≈0,2 %). La mediana confirma la dirección (+7,6 % en A, +4,5 % en B, +7,3 % en C); en C el promedio casi no se mueve pero la mediana sí, lo que sugiere que la subida se da en los departamentos típicos y no en los de mayor volumen. _(Cifras de la corrida local con los mismos CSV; verificar contra la salida en Databricks.)_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q2 · Función de ventana: crecimiento semana a semana y ranking de semanas por tienda (P1)
# MAGIC Venta total semanal por tienda, variación frente a la semana anterior (`LAG`) y posición de cada semana dentro de su tienda (`RANK`).

# COMMAND ----------

q2 = spark.sql("""
WITH semanal AS (
  SELECT Store, Date, MAX(CAST(IsHoliday AS INT)) AS festivo, SUM(Weekly_Sales) AS venta_total
  FROM v GROUP BY Store, Date
)
SELECT Store, Date, festivo, ROUND(venta_total, 2) AS venta_total,
       ROUND(100 * (venta_total - LAG(venta_total) OVER (PARTITION BY Store ORDER BY Date))
             / LAG(venta_total) OVER (PARTITION BY Store ORDER BY Date), 2) AS var_pct_semana_previa,
       RANK() OVER (PARTITION BY Store ORDER BY venta_total DESC) AS ranking_en_tienda
FROM semanal
""")
q2.createOrReplaceTempView("q2")
display(spark.sql("SELECT festivo, ROUND(AVG(var_pct_semana_previa),2) AS var_prom_pct, COUNT(*) AS semanas FROM q2 GROUP BY festivo"))
display(spark.sql("SELECT Store, Date, festivo, venta_total FROM q2 WHERE ranking_en_tienda = 1 ORDER BY venta_total DESC LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación Q2.** El crecimiento semanal promedio es 1,78 % en semanas marcadas como festivas frente a 0,58 % en las demás. Las diez semanas más altas del ranking son casi todas la del 24-dic-2010, que **no está marcada** como festiva en `IsHoliday` (la marca cae en la semana del 31-dic). Esto indica que la bandera del dataset subestima el pico real de Navidad y no debería usarse sola para planear inventario.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3 · Join entre 2 fuentes: venta por tipo y tamaño de tienda, y venta por departamento (P2)
# MAGIC Une el resumen de ventas (fuente `train`) con el catálogo de tiendas (fuente `stores`).

# COMMAND ----------

q3 = spark.sql("""
SELECT t.Type AS tipo, COUNT(DISTINCT s.Store) AS tiendas,
       ROUND(SUM(s.venta), 0) AS venta_total,
       ROUND(SUM(s.venta) / SUM(t.Size) , 2) AS venta_por_unidad_de_tamano
FROM (SELECT Store, SUM(Weekly_Sales) AS venta FROM v GROUP BY Store) s
JOIN t ON s.Store = t.Store
GROUP BY t.Type ORDER BY venta_total DESC
""")
display(q3)
display(spark.sql("""
SELECT Dept, ROUND(SUM(Weekly_Sales),0) AS venta_total,
       ROUND(100*SUM(Weekly_Sales)/SUM(SUM(Weekly_Sales)) OVER (), 2) AS pct_del_total
FROM v GROUP BY Dept ORDER BY venta_total DESC LIMIT 10"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación Q3.** Las 22 tiendas A suman ≈4.331 M, las 17 B ≈2.001 M y las 6 C ≈405 M. Por unidad de tamaño las C rinden más (≈1.667 frente a ≈1.111 en A), es decir, venden más por unidad de tamaño. Los departamentos 92, 95 y 38 concentran ≈19,7 % de la venta total; el top-10 llega a ≈46 %. _(Verificar contra Databricks.)_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q4 · Correlación de la venta semanal con clima, combustible, CPI y desempleo (P3)

# COMMAND ----------

q4 = spark.sql("""
WITH s AS (
  SELECT Store, Date, SUM(Weekly_Sales) AS venta, AVG(Temperature) AS temp, AVG(Fuel_Price) AS comb,
         AVG(CPI) AS cpi, AVG(Unemployment) AS desempleo
  FROM v GROUP BY Store, Date)
SELECT ROUND(CORR(venta, temp),3) AS corr_temperatura, ROUND(CORR(venta, comb),3) AS corr_combustible,
       ROUND(CORR(venta, cpi),3) AS corr_cpi, ROUND(CORR(venta, desempleo),3) AS corr_desempleo
FROM s""")
display(q4)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación Q4.** Las correlaciones con la venta total por tienda-semana son débiles: temperatura −0,064, combustible 0,009, CPI −0,073 y desempleo −0,106. Mezclar tiendas de tamaños muy distintos oculta el efecto; una correlación baja aquí no descarta estas variables. Para la Fase 3 conviene medirla dentro de cada tienda o normalizar por tamaño.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q5 · Efecto de la intensidad de promoción × festivo desde noviembre de 2011 (P3)
# MAGIC Solo desde `2011-11-11`, la primera semana con MarkDown en el dataset. Desde ahí casi todas las tienda-semana tienen alguna
# MAGIC promoción, así que comparar "con" contra "sin" promoción es poco informativo; se compara por **intensidad**
# MAGIC (suma de MarkDown1-5: sin promoción / baja / alta, partiendo en la mediana de las semanas con promoción).

# COMMAND ----------

q5 = spark.sql("""
WITH base AS (
  SELECT Weekly_Sales, IsHoliday, MarkDown1+MarkDown2+MarkDown3+MarkDown4+MarkDown5 AS md_total
  FROM v WHERE Date >= DATE'2011-11-11'),
corte AS (SELECT PERCENTILE_APPROX(md_total, 0.5) AS mediana FROM base WHERE md_total > 0)
SELECT CASE WHEN md_total <= 0 THEN '1 sin promoción'
            WHEN md_total <= mediana THEN '2 promoción baja' ELSE '3 promoción alta' END AS intensidad,
       IsHoliday AS festivo, COUNT(*) AS registros, ROUND(AVG(Weekly_Sales), 2) AS venta_promedio
FROM base CROSS JOIN corte
GROUP BY 1, 2 ORDER BY 1, 2""")
display(q5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación Q5.** Desde 11-nov-2011 no hay semanas sin promoción: todas tienen alguna. Con promoción alta la venta promedio es ≈19.300 frente a ≈13.200 con promoción baja en semanas normales. En festivo con promoción baja la venta cae a ≈9.760 (solo 2.399 registros), mientras que con promoción alta se mantiene ≈19.630. Es asociación, no causalidad: las promociones altas pueden concentrarse en tiendas grandes.
