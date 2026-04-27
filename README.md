# Speed Estimation y Centroid Tracking

## Descripción
- Este repositorio contiene el desarrollo de un sistema que estima la velocidad de vehículos utilizando *Centroid Tracking* para el seguimiento de objetos, y el modelo Yolov8n para la detección.

- Asimismo, se aplica la transformación de perspectiva para convertir el centroide detectado a una escala que se corresponde con el plano de la carretera y estimar la velocidad de forma menos sesgada.

## Visualización
<div align="center">
    <img src="./TEST/SpeedEstimationFPS.gif">
</div>

## Archivos
- [main.py](./main.py):
    - Script principal que une las funciones de los archivos para estimar la velocidad de los vehículos y realizar la visualización.
- [centroid_tracking.py](./centroid_tracking.py):
    - Script que contiene el algoritmo *Centroid Tracking* implementado *desde cero*. 
- [transform_perspective.py](./transform_perspective.py):
    - Script que calcula la matriz de transformación de perspectiva y convierte los puntos del centroide del objeto a la nueva escala.
- [utils.py](./utils.py):
    - Script que tiene la extracción de bounding boxes y su visualización.
