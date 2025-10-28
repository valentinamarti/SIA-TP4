# SIA TP4 - Aprendizaje No Supervisado 🧠

Este trabajo práctico tiene como objetivo implementar tres modelos clave de Aprendizaje No Supervisado:

1.  **Red de Kohonen**: Utilizada para *clustering* de países en el Ejercicio Europa.
2.  **Regla de Oja**: Aplicada para el cálculo de la **Primera Componente Principal (PC1)** en el Ejercicio Europa.
3.  **Modelo de Hopfield**: Empleado como **memoria asociativa** para el almacenamiento y recuperación de patrones en el Ejercicio Patrones.

---
## Prerrequisitos
- [Python](https://www.python.org/downloads/) instalado en el sistema.
- `pip` disponible en la terminal (`pip --version` para verificar).

---
## Construcción
Para construir el proyecto por completo y contar con el entorno necesario, ejecute de manera secuencial los siguientes comandos desde la raíz:

### Windows:
```bash

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/MacOS
```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

---
## Comandos de Ejecución

Una vez que el entorno virtual esté activado y las dependencias instaladas, puedes ejecutar cada parte del trabajo 
práctico utilizando el *script* principal `python main.py` junto con los argumentos correspondientes.


### 1. Ejercicio Europa (`europe.csv`)
Este ejercicio aborda la reducción de dimensionalidad y *clustering* sobre el *dataset* de países europeos (en ./data/europe.csv).

#### 1.1. Red de Kohonen
Implementación de la Red de Kohonen para realizar *clustering*. Genera múltiples gráficos de resultados que se guardan en la carpeta `./results`.

| Tarea | Comando de Ejecución (Ejemplo)           |
| :--- |:-----------------------------------------|
| **Generar Mapa y Gráficos** | python main.py kohonen ./data/europe.csv |


#### 1.2. Regla de Oja (PCA)
Implementación de la Regla de Oja para calcular la Primera Componente Principal (PC1) y su comparación con una implementación de librería.

| Tarea                                          | Comando de Ejecución (Ejemplo)                 |
|:-----------------------------------------------|:-----------------------------------------------|
| **Calcular PC1 con libreria externa**          | python main.py pca ./data/europe.csv           |
| **Calcular PC1 con Oja**                       | python main.py oja ./data/europe.csv           |
| **Calcular y Comparar PC1 (Oja vs. Librería)** | python main.py pca ./date/europe.csv --compare |

---

### 2. Ejercicio Patrones
Este ejercicio utiliza el **Modelo de Hopfield** como memoria asociativa con patrones de letras $5\times5$ (`1` y `-1`).

#### 2.1. Modelo de Hopfield

| Tarea | Comando de Ejecución (Ejemplo) |
| :--- |:-------------------------------|
| **Asociación de Patrones** (Convergencia) | python main.py hopfield        |
| **Estado Espúreo** (Patrón Ruidoso) | python main.py hopfield        |