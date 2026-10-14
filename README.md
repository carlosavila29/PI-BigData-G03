# Proyecto Integrador Big Data · Grupo 03 · Retail (Walmart Store Sales Forecasting)

Uniciencia · Big Data · Docente: Ing. Jacksson Sonny Gonzalez Bayona · Herramienta: Databricks Free Edition.

## Estructura (protocolo v2)
| Carpeta | Contenido | Fase |
|---|---|---|
| `00_ingesta` | Capa bronce (4 CSV a tablas Delta, 3 validaciones) | 1 |
| `01_limpieza` | Capa plata: 6 transformaciones y bitácora | 2 |
| `02_analisis` | 5 consultas (ventana y join) y optimización con tiempos | 2 |
| `03_seguridad` | Clasificación, anonimización y control de acceso | 2 |
| `04_visualizacion` | Capa oro y tablero | 3 (pendiente) |
| `99_evidencias` | Informe, actas y capturas | 1-3 |

## Cómo reproducir
1. Descargar los 4 CSV de https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data (descarga registrada: 6 de septiembre de 2026; Kaggle exige sesión, por lo que se usó un espejo público en GitHub, `vicky60629/Walmart-Store-Sales-Forecasting`, con el mismo contenido).
2. Subirlos a `/Volumes/workspace/default/proyecto_integrador_grupo03/` (`train.csv` se subió comprimido como `train.csv.gz`).
3. Importar los notebooks `.py` en Databricks y correrlos en orden: `00_ingesta`, `01_limpieza`, `02_analisis`, `03_seguridad`, `02b_optimizacion`.

## Estado
Fase 2: notebooks ejecutados en Databricks el 14 de octubre de 2026. Pendiente: acta de auditoría cruzada (A2.7).
