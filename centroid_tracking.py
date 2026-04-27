#Centroid tracking algorithm
import cv2
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self,max_frames_delete,threshold_distance):
        self.id=0
        self.max_frames_delete=max_frames_delete
        self.threshold_distance=threshold_distance

        self.detected_objects={}
        self.frames_delete={}

    def add_object(self,centroid):
        self.detected_objects[self.id]=centroid
        self.frames_delete[self.id]=0
        self.id+=1


    def delete_object(self,object_id):
        del self.detected_objects[object_id]
        del self.frames_delete[object_id]


    def update_frame(self,boxes,frame): #[frame no]
        #Si NO hay boxes
        if len(boxes)==0:
            for id_obj in self.detected_objects.keys():
                self.frames_delete[id_obj]+=1

                if self.frames_delete[id_obj]>=self.max_frames_delete:
                    self.delete_object(id_obj)

            return self.detected_objects
    
      
        #Si sí hay boxes: Este calcula el centroide en el orden de los bounding boxes
        centroids_arrays=np.zeros(shape=(len(boxes),2))
        for (i,(x1,y1,w,h)) in enumerate(boxes):
            x2,y2=x1+w,y1+h
            x_c,y_c=(x1+x2)/2,(y1+y2)/2
            centroids_arrays[i][0],centroids_arrays[i][1]=x_c,y_c
            cv2.circle(img=frame,center=(int(x_c),int(y_c)),radius=2,color=(0,255,0),thickness=2)

        #Si no hay objetos detectados aún
        if len(self.detected_objects)==0:
            for i in range(len(boxes)):
                #print(f"objeto {i+1} detectado")
                self.add_object(centroids_arrays[i])
            
                
        #OJO: Aquí los objetos NUEVOS NO son añadidos, ya que los objetos que se guardan solo son los que se detectan
        #en el frame inicial, todo por la condicional de arriba --> Solucionarlo
        #Además, ver si en cada frame se añade algo nuevo, ¿el bounding box del objeto 0 tendrá igual índice que en el frame 20?
        
        prev_id_objects=list(self.detected_objects.keys())
        prev_centroids=list(self.detected_objects.values())
       

        distancias=dist.cdist(XA=np.array(prev_centroids),XB=centroids_arrays)
        


        #Evitar duplicados de tracking
        rows=distancias.min(axis=1).argsort()       #rows (filas) de los objetos anteriores ordenados de menor a mayor (más fácil a más difícil)
        columns=distancias.argmin(axis=1)[rows]     #index objects ordenados de menor a mayor distancia
        
        rows_used=set()
        cols_used=set()
        for (row,col) in zip(rows,columns):
            if col in cols_used:
                continue            #Aquí me aseguro de que NO hayan objetos duplicados

            id_object=prev_id_objects[row]

            #Actualización del centroide SOLO si es menor al threshold
            distancia_actual=np.linalg.norm(self.detected_objects[id_object]-centroids_arrays[col])
            if distancia_actual>self.threshold_distance:
            #    print(f"distancia actual: {distancia_actual}")
            #    print(f"id object: {id_object}")
                continue

            self.detected_objects[id_object]=centroids_arrays[col]    #Actualización del centroide
            
            
            #self.frames_delete[id_object]=0                           #Inicializo los valores para evitar key error
          
            rows_used.add(row)
            cols_used.add(col)


      
        #Notar que sobran algunos objetos --> en un frame pueden haber 3, en el sig. frame 5. Esos 2 restantes deben ser añadidos como obj nuevos
        rows_no_used=set(range(0,distancias.shape[0])).difference(rows_used)    #objects nuevos
        cols_no_used=set(range(0,distancias.shape[1])).difference(cols_used)    
       

        #SÍ THRESHOLD
        #Si hay objetos antiguos que NO se usan
        for row in rows_no_used:
            id_object=prev_id_objects[row]
            self.frames_delete[id_object]+=1

            if self.frames_delete[id_object]>self.max_frames_delete:
                self.delete_object(id_object)

        #Si hay objetos nuevos
        for col in cols_no_used:
            self.add_object(centroid=centroids_arrays[col])

    
        return self.detected_objects
