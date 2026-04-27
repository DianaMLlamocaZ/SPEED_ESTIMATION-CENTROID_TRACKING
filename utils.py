import torch
import numpy as np
import cv2

#Función que devuelve los bboxes y scores para realizar un filtrado en NMS
def get_boxes_scores(results): #[1,84,8400]
    results_tensor=torch.tensor(results).permute(0,2,1) #[84] --> [0:4] bboxes, [4:] scores, [1,8400,84]
  
    bboxes,scores=results_tensor[:,:,0:4].squeeze(0),results_tensor[:,:,4:].squeeze(0)
    bboxes[:,0],bboxes[:,1]=bboxes[:,0]-bboxes[:,2]/2,bboxes[:,1]-bboxes[:,3]/2 #[x_center,y_center,w,h] -->[x1,y1,w,h]; x1,y1 upon left corner
  
    prob_score,index=scores.max(dim=1)

    list_bboxes,list_scores=[bbox.tolist() for bbox in bboxes],[score.item() for score in prob_score]
    list_clases=[clase for clase in index]
    

    return list_bboxes,list_scores,list_clases



#Función que dibuja los rectángulos
def draw_bbox(bboxes_list,clases_list,frame):
    for index,bbox in enumerate(bboxes_list):
        x_1,y_1,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
        cv2.rectangle(img=frame,pt1=(int(x_1),int(y_1)),pt2=(int(x_1+w),int(y_1+h)),color=(0,0,255),thickness=1)
