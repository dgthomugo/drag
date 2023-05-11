import vega
import os
import numpy as np


class Ray3D:
    def __init__(self, camModel, model_path):
        iface = vega.get_interface("RayThreeD")
        if iface is not None:
            vega.destroy_interface(iface)
        if not os.path.exists(model_path):
            raise Exception(f"RayThreeD model directory {model_path} not exist")
        self.model = vega.Model(model_path, create=True)
        self.camera = camModel
        self.receptive_field = 9
        self.pad = (self.receptive_field - 1) // 2  # Padding on each side
        self.poseData = {}
        self.input_param = np.array([-self.camera.Tc2n[1,0], self.camera.cam_pitch_rad], dtype='float32').reshape(1, 2, 1, 1)

    def extract_keypoints3d(self, poseDic, frameIndex):
        result_trackId = []
        input_ray_all = None
        for k, v in poseDic.items():
            result_trackId.append(k)
            cur_rayinput = self.camera.getRayInput(v[..., :2].astype(np.float32))

            if k in self.poseData:
                pose2dRayArray = self.poseData[k]["point2dRay"]
                pose2dRayArray = np.concatenate((pose2dRayArray, cur_rayinput), axis=0)
                self.poseData[k]["point2dRay"] = pose2dRayArray[-self.pad:]
                self.poseData[k]["curFrameIndex"] = frameIndex
            else: 
                pose2dRayArray = cur_rayinput
                self.poseData[k] = {"point2dRay": pose2dRayArray, "curFrameIndex": frameIndex}

            input_keypoints = pose2dRayArray.copy()
            pad_r = self.pad
            pad_l = self.pad - pose2dRayArray.shape[0] + 1
            input_rays = np.expand_dims(np.pad(input_keypoints, ((pad_l, pad_r), (0, 0), (0, 0)), 'edge'), axis=0)

            if input_ray_all is None:
                input_ray_all = input_rays
            else:
                input_ray_all = np.concatenate((input_ray_all, input_rays), axis=0)

        param = np.repeat(self.input_param, len(input_ray_all), axis=0)
        data = input_ray_all.astype('float32')
        err, rsp = self.model.infer([data[0], param[0]], bypass_prep=True)
        assert err is None
        root = None
        pose = None
        for item in rsp.get_all_items():
            tag = item.tag
            if tag == 908001002:
                root = np.array(item.get_floats(vega.DataKind.FEATURE), dtype=np.float32)
                #print(f'root: {root.reshape(1,1,3,1)}')
            else:
                pose = np.array(item.get_floats(vega.DataKind.FEATURE), dtype=np.float32)
                #print(f'pose: {pose.reshape(1,24,3,1)}')

        pose3d_result = root.reshape(1,1,3,1) + pose.reshape(1,24,3,1)
        #print(f'result: {pose3d_result}')
        self.updatePoseData(frameIndex)
        return result_trackId, pose3d_result

    def normalize_screen_coordinates(self, X):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / self.res_w * 2 - [1, self.res_h / self.res_w]

    def updatePoseData(self, frameIndex):
        poseDataTemp = self.poseData
        self.poseData = {}
        for k, v in poseDataTemp.items():
            if frameIndex - v["curFrameIndex"] < 50:
                self.poseData[k] = v
