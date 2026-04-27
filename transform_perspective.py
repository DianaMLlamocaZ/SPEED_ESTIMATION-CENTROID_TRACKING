import numpy as np
import cv2


class TransformPoints:
    def __init__(self,src_points,trg_points):
        self.src_points=src_points
        self.trg_points=trg_points
        self.matrix=cv2.getPerspectiveTransform(src=self.src_points,dst=trg_points)

    def transform(self,bbox_points):
        bbox_reshaped=np.array(bbox_points,dtype=np.float32).reshape(-1,1,2)
        print(f"bbox reshaped: {bbox_reshaped.shape}")
        warped_bbox=cv2.perspectiveTransform(src=bbox_reshaped,m=self.matrix)

        return warped_bbox.reshape(-1)
