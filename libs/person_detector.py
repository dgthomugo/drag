import vega
import os
import numpy as np


class PersonDetector:
    def __init__(self, model_path):
        iface = vega.get_interface("PersonDetAISports")
        if iface is not None:
            vega.destroy_interface(iface)
        if not os.path.exists(model_path):
            raise Exception(f"person detect model directory {model_path} not exist")
        self.model = vega.Model(model_path, create=True)
        
    def detect(self, image):
        err, rsp = self.model.infer(image)
        assert err is None
        boxes = rsp.as_detection().boxes
        result = []
        for box in boxes:
            result.append([2, 0.9, int(box.x), int(box.y), int(box.x+box.w), int(box.y+box.h)])
        return np.array(result)
