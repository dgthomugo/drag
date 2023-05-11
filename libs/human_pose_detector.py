import vega
import os
import cv2
import numpy as np

    
class HumanPoseDetector:
    def __init__(self, model_path: str):
        iface = vega.get_interface("HumanPose")
        if iface is not None:
            vega.destroy_interface(iface)
        if not os.path.exists(model_path):
            raise Exception(f"human pose model directory {model_path} not exist")
        self.model = vega.Model(model_path, create=True)  
          
    # box{x1, y1, x2, y2}
    def get_keypoints(self, image: cv2.Mat, bbox):
        box = vega.BBox()
        box.x = int(bbox[0])
        box.y = int(bbox[1])
        box.w = int(bbox[2]-bbox[0])
        box.h = int(bbox[3]-bbox[1])
        err, rsp = self.model.infer(image, roi=box)
        assert err is None
        for item in rsp.get_all_items():
            kps = np.array(item.get_floats(vega.DataKind.KEYPOINTS)).reshape(-1, 2)[:24]
            score = np.array(item.get_floats(vega.DataKind.CONFIDENCES)).reshape(-1, 1)[:24]
            return np.concatenate([kps, score], axis=1)[np.newaxis, ...]
        return None
        