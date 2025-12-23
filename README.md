# 🔥 Repositorio de la Tesis de Maestría en Ciencias Físicas del Instituto Balseiro: Simulaciones computacionales y visualización de la propagación de incendios forestales en la región patagónica

## Maestrando: Lic. Lucas Becerra
## Directora: Dra. Karina Laneri
## Co-directora: Dra. Mónica Malen Denham

Este repositorio contiene el material correspondiente a la Tesis de Maestría en Ciencias Físicas del Instituto Balseiro, titulada 'Simulaciones computacionales y visualización de la propagación de incendios forestales en la región patagónica', dirigida por la Dra. Karina Laneri y por la Dra. Mónica Malen Denham.

## 📁 Estructura del repositorio

- `modelo_rdc.py` — Implementación del modelo de reacción-difusión-convección.

La implementación numérica del modelo RDC fue realizada mediante descomposición de operadores (*operator splitting*) en la que los términos de reacción y convección fueron discretizados mediante un esquema de Euler explícito y el término de difusión mediante el esquema implícito *alternating direction implicit* (ADI). La descripción matemática de los métodos se encuentra en el Capítulo 2 de la tesis.

- `fuego_referencia.py` — Simulación de referencia para comparación entre métodos.

El programa `fuego_referencia.py` permite realizar simulaciones de referencia con distintos parámetros. Se encuentran configurados los tres experimentos sintéticos descritos en la tesis y utilizados para recuperar los parámetros. Para correr cualquiera de los tres experimentos sintéticos:

```bash
python fuego_referencia.py --exp 1 --visualizar_mapas
```

- `mapas/`
 |
 | - `mapas_steffen_martin` - Contiene los mapas raster utilizados
 | - `io_mapas.py` - Funciones de lectura y procesado de mapas

- `genetico/` - Contiene los scripts en python para ejecutar los métodos de fuerza bruta y el algoritmo genético
 |
 | - `algoritmo.py` - Itera el algoritmo genético utilizando los operadores evolutivos
 | - `config.py` - Contiene valores como el tamaño del paso temporal, la distancia entre celdas
 | - `fitness.py` - Realiza una simulación con una configuración dada de parámetros y calcula el fitness de cada configuración
 | - `lectura_datos.py` - Carga una población entreenada y se encarga de guardarla luego de una corrida del algoritmo genético
 | - `operadores_geneticos.py` - Contiene la implementación de los operadores de selección, cruce y mutación
 | - `main.py` - Ejecuta el algoritmo genético. Corre cualquiera de los tres experimentos sintéticos descritos en la Tesis. Para correrlo se requiere el mapa de referencia generado por `fuego_referencia.py`. Para correrlo desde el directorio Genetico, hay que ejecutar: 
```bash
main.py --exp 1
```
 | - `fuerza_bruta.py` — Exploración de parámetros por búsqueda exhaustiva (brute force). Para correrlo desde el directorio Genetico, hay que ejecutar:
  ```bash
  fuerza_bruta.py --exp 1
  ```

- `pinns/` — Entrenamiento de redes neuronales informadas por la física (Physics-Informed Neural Networks, PINNs).
 |
 | - `train_pinn.py` - Modelo de PINN
 | - `pinns_sir.py` - Entrenamiento de la PINN. Para entrenar una PINN desde el directorio PINNS, hay que ejecutar:
 ```bash
 python pinns_sir.py
 ```

- `.gitignore` — Ignora archivos temporales y entornos virtuales, de Python.
- `README.md` — Este archivo.

## ⚙️ Dependencias y requerimientos

El código de este repositorio fue desarrollado en **Python** y está orientado a la simulación numérica y análisis computacional de incendios forestales, con énfasis en ejecución acelerada por GPU.

### 📦 Dependencias principales

Las principales bibliotecas utilizadas son:

- **NumPy** — Operaciones numéricas y manejo de arreglos.
- **SciPy** — Métodos numéricos y resolución de sistemas.
- **Matplotlib** — Visualización de resultados.
- **CuPy** — Computación acelerada por GPU compatible con CUDA.
- **PyTorch** — Implementación y entrenamiento de redes neuronales informadas por la física (PINNs).
- **Rasterio** — Lectura y manejo de mapas raster geoespaciales.

Algunas dependencias pueden ser opcionales dependiendo del módulo que se desee ejecutar.

### 🚀 Requerimientos de GPU

- GPU compatible con **CUDA** (NVIDIA).
- Drivers de NVIDIA y versión de CUDA compatibles con la versión de **CuPy** instalada.
- Para el entrenamiento de PINNs, se recomienda disponer de al menos **8 GB de memoria de GPU**.