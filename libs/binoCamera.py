import json
import cv2
import numpy as np

IMG_W = 1920
IMG_H = 1080

class BinoCamera:
    def __init__(self, camParamFile) -> None:
        json_data = json.load(open(camParamFile, "r"))["cameras"]
        K1 = np.array(json_data[0]["kmat"], dtype=np.float64).reshape(3, 3)
        K2 = np.array(json_data[1]["kmat"], dtype=np.float64).reshape(3, 3)

        D1 = np.array(json_data[0]["dvec"], dtype=np.float64).reshape(5, 1)
        D2 = np.array(json_data[1]["dvec"], dtype=np.float64).reshape(5, 1)
        RT = np.array(json_data[1]["RT"], dtype=np.float64).reshape(3, 4)
        # R = RT[:, :3]
        # T = RT[:, 3]

        # R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (1920, 1080), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.85)
        R1 = np.array(json_data[0]["R1"], dtype=np.float64).reshape(3, 3)
        P1 = np.array(json_data[0]["P1"], dtype=np.float64).reshape(3, 4)
        R2 = np.array(json_data[1]["R2"], dtype=np.float64).reshape(3, 3)
        P2 = np.array(json_data[1]["P2"], dtype=np.float64).reshape(3, 4)

        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)
        self.leftP = P1
        self.rightP = P2
        self.f = P1[0, 0]
        self.d = np.abs(P2[0, 3])
        self.cx = P1[0, 2]
        self.cy = P1[1, 2]

        self.cam2GroundRT = np.array([[0.07355645412619927, 0.9955067163245126, -0.05963074550235248, 0.6685360658344782],
                             [0.3638000786112687, -0.08245695745858295, -0.9278202158657183, 1.4035834301574807],
                             [-0.9285682262810928, 0.046553495244282705, -0.3682306630655921, 5.698079012436084]],dtype=np.float32)

    def remapImage(self, imgL, imgR):
        imgMapL = cv2.remap(imgL, self.leftMapX, self.leftMapY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        imgMapR = cv2.remap(imgR, self.rightMapX, self.rightMapY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return imgMapL, imgMapR

    def triangle(self, poseL, poseR):
        Point_3D = cv2.triangulatePoints(self.leftP, self.rightP, poseL.T, poseR.T)
        pnt3d = Point_3D / Point_3D[3]
        return pnt3d[:3]

    def reconstruct3dpose(self, pose2dArr, disparityArr):
        assert len(pose2dArr) == len(disparityArr)
        pose3dList = []
        for index in range(len(pose2dArr)):
            pose2d = pose2dArr[index]
            disparity = disparityArr[index]
            z = self.f * self.d / (disparity + 1e-5)
            z = z * (disparity > 5)
            x = (pose2d[0, :, 0] - self.cx) / self.f
            x = x * z
            y = (pose2d[0, :, 1] - self.cy) / self.f
            y = y * z
            visable = (disparity > 5) * pose2d[0, :, 2]
            pose3d = np.concatenate([x, y, z]).reshape(3, -1)
            pose3d = pose3d / 1000    # mm to m
            pose3d_ground = np.dot(self.cam2GroundRT[:,:3].T, pose3d - self.cam2GroundRT[:,3:4])
            pose3dList.append(np.concatenate([pose3d_ground, visable.reshape(1,-1)], axis=0))
        return pose3dList

    def getDisparityFromPose3d(self, pose3dArray):
        '''
        pose3dArray: shape (3,24)
        '''
        disparity = self.f * self.d / (pose3dArray[2, :] * 1000 + 1e-5)
        disparity = disparity * (disparity < IMG_W)
        return disparity
