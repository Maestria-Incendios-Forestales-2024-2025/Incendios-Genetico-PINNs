# 🔥 Incendios-Forestales---MCF-2024-2025

Este repositorio contiene el material correspondiente a la Tesis de Maestría en Ciencias Físicas del Instituto Balseiro, titulada 'Simulaciones computacionales y visualización de la propagación de incendios forestales en la región patagónica', dirigida por la Dra. Karina Laneri y por la Dra. Mónica Malen Denham.

## 📁 Estructura del repositorio

- `modelo_rdc.py` — Implementación del modelo de reacción-difusión-convección.

La implementación numérica del modelo RDC fue realizada mediante descomposición de operadores (*operator splitting*) en la que los términos de reacción y convección fueron discretizados mediante un esquema de Euler explícito y el término de difusión mediante el esquema implícito *alternating direction implicit* (ADI). La descripción matemática de los métodos se encuentra en el Capítulo 2 de la tesis.

- `fuego_referencia.py` — Simulación de referencia para comparación entre métodos.

El programa `fuego_referencia.py` permite realizar simulaciones de referencia con distintos parámetros. Se encuentran configurados los tres experimentos sintéticos descritos en la tesis y utilizados para recuperar los parámetros. Para correr cualquiera de los tres experimentos sintéticos:

```bash
python fuego_referencia.py --exp 1 --visualizar_mapas
```

- `gif_simulacion.py` permite realizar un archivo .gif de una simulación con parámetros dados, modificados internamente en el programa. 

- `fuerza_bruta.py` — Exploración de parámetros por búsqueda exhaustiva (brute force).
- `algoritmo_genetico.py` — Implementación de un algoritmo genético para ajuste de parámetros.
- `PINNS/` — Entrenamiento de redes neuronales informadas por la física (Physics-Informed Neural Networks, PINNs).
- `.gitignore` — Ignora archivos temporales y entornos virtuales, de Python.
- `README.md` — Este archivo.
