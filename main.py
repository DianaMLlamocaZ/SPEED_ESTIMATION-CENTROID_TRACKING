import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#from ultralytics import YOLO
from utils import get_boxes_scores,draw_bbox
from centroid_tracking import CentroidTracker
from transform_perspective import TransformPoints
from collections import deque


#Modelo: Uncomment para descargar el modelo
#modelo=YOLO("yolov8n.pt")
#modelo.export(format="onnx",imgsz=[640,640])

#Cargar el modelo
mod=cv2.dnn.readNet(model="./yolov8n.onnx")


#Cargar el video
video=cv2.VideoCapture(filename="./data/vehicles.mp4")


#Obtengo los fps del video
fps_video=video.get(cv2.CAP_PROP_FPS)

#Tracker
ct=CentroidTracker(max_frames_delete=15,threshold_distance=50)


#Historial de seguimiento: coordenadas p/ frames
historial_coords={}


#Velocidad seguimiento
velocidad_tracking={}


#Guardar video
fourcc=cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out=cv2.VideoWriter('./data_results/video_proc.mp4', fourcc, 15, (640, 640))


#Src and Target points
#Source points image
SOURCE=np.array([
    [216,225],
    [382.1,234.6],
    [750,640],
    [-150,640]
],dtype=np.float32)

TARGET=np.array([
    [0,0],
    [25,0],
    [25,250],
    [0,250]
],dtype=np.float32)


#Transform points
transform_points=TransformPoints(src_points=SOURCE,trg_points=TARGET)

frame_rsz=cv2.resize(src=video.read()[1],dsize=(640,640))   #(640,640)
matrix=cv2.getPerspectiveTransform(src=SOURCE,dst=TARGET)



#Bucle principal
while True:
    ret,frame=video.read()

    #Preprocesamiento de imgs
    frame_rsz=cv2.resize(src=frame,dsize=(640,640))
    blob_img=cv2.dnn.blobFromImage(image=frame_rsz,scalefactor=1/255,swapRB=True)
    


    #Predicción
    mod.setInput(blob=blob_img)
    preds=mod.forward()


    #Threshold y NMS
    bboxes,scores,clases=get_boxes_scores(results=preds)        #bboxes: [x1,y1,w,h]
    indexes_bboxes=cv2.dnn.NMSBoxes(bboxes=bboxes,scores=scores,score_threshold=0.5,nms_threshold=0.3)
    
    bboxes_selected=[bboxes[index] for index in indexes_bboxes]
    clases_selected=[clases[index] for index in indexes_bboxes]
    

    #Object tracking --> Obtengo los centroides de cada objeto
    ct.update_frame(boxes=bboxes_selected,frame=frame_rsz)
    objetos=ct.detected_objects
    
    print(f"objetos actuales: {objetos}")

    #Mapeo los centroides a la nueva perspectiva
    for id in objetos:
        object_centroid=objetos[id] #Centroid escala org
        transformed_point=transform_points.transform(bbox_points=object_centroid)   #Centroid escala perspectiva


        #Dibujar el ID del objeto cuando es detectado
        cv2.putText(frame_rsz,f"ID: {id}",(int(object_centroid[0]),int(object_centroid[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        

        #Añadir el id del objeto para calcular la velocidad (si no está aún en el diccionario)
        if id not in historial_coords:
            historial_coords[id]=deque(iterable=[],maxlen=20) #maxlen=int(fps_video)
            historial_coords[id].append(transformed_point)
            

        #Si ya está, solo almacenarlo en la memoria
        else:
            historial_coords[id].append(transformed_point)
            
        

        #Calcular velocidad SOLO si el objeto tiene 20 coords trackeadas --> Para evitar que la velocidad se dispare frame a frame al inicio
        if len(historial_coords[id])==20:   #>=int(fps_video)//2
            
            #Calculo la velocidad promedio en base a la coordenada inicial en memoria, y la coordenada final actual
            coord_init,coord_fnl_actual=historial_coords[id][0],historial_coords[id][-1]
            

            #Calculo la distancia
            distancia=np.linalg.norm(abs(coord_fnl_actual-coord_init))

            #Calculo el tiempo
            tiempo=len(historial_coords[id])/fps_video
            

            #Velocidad
            velocidad=(distancia/tiempo)
            velocidad_tracking[id]=velocidad    #Almaceno la velocidad en un diccionario para el tracking del objeto

            
            #Dibujar la velocidad
            cv2.putText(frame_rsz, f"{round(velocidad,2)} m/s", (int(object_centroid[0]-25),int(object_centroid[1])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        


    #Dibujando los bboxes
    draw_bbox(bboxes_list=bboxes_selected,clases_list=clases_selected,frame=frame_rsz)


    cv2.imshow(winname="Video",mat=frame_rsz)
    out.write(image=frame_rsz)

    img_transformed=cv2.warpPerspective(src=frame_rsz,M=matrix,dsize=(32,140))
    cv2.imshow(winname="transformed",mat=img_transformed)
    

    tecla=cv2.waitKey(1)
    if tecla==ord("q") or not ret:
        break
