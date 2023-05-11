from operator import mul
import math
import numpy as np
import cv2

def dotproduct(v1, v2):
    return sum(map(mul, v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


class Camera:
    
    def __init__(self, K, dist, R=None, t=None, camHeight=None, campitch=None):
        """
        :param K: intrinsic matrix, (3, 3)
        :param dist: distortion coefficient, (5,)
        :param camHeight: camera height (1,) m
        :param campitch:  camera pitch (1,) rad
        :param R: rotation matrix, (3, 3)
        :param t: translation vector, (3, 1)
        """
        self.K = K.astype(np.float64)
        self.invK = np.linalg.inv(self.K)
        self.dist_coeff = dist.astype(np.float64)

        if not (R is None or t is None):
            self.Rw2c = R.astype(np.float64)
            self.Tw2c = t.astype(np.float64)
            self.Rc2w = self.Rw2c.T
            self.Tc2w = -self.Rw2c.T @ self.Tw2c
            self.cam_ray_world = self.get_cam_ray_world()
            self.cam_pitch_rad = self.get_cam_pitch_rad()
            cam_orig_world = -self.Rw2c.T @ self.Tw2c
            self.camHeight = cam_orig_world[2]
        elif camHeight and campitch:
            self.camHeight = camHeight
            self.cam_pitch_rad = campitch
        else:
            print("(R t) and  (camHeight,campitch) must provide one group!")

        self.Rc2n, self.Tc2n = self.get_norm_coord_config()

    def get_cam_ray_world(self):
        """
        return the ray of camera in world coordinate system
        # define the vector that starts from camera center to principal(focal) point as representation of the camera.
        # suppose that the focal point is normalized,
        # we convert the vector to world space to represent the ray of the camera.
        :return:
        """
        focal_pt_cam = np.asarray([0, 0, 1], np.float64)
        P_w = self.Rc2w @ focal_pt_cam
        return P_w[0:3].reshape((3, 1))

    def get_cam_pitch_rad(self):
        """
        return camera pitch in radius
        # here we assume the camera is looking towards to the ground
        :return:
        """
        ray_upright = np.zeros((3, 1)).astype(np.float64)
        ray_upright[2] = 1
        return angle(self.cam_ray_world, ray_upright) - np.pi / 2

    def get_norm_coord_config(self):
        """
        rotate the camera about the x-axis to eliminate the pitch.
        in normalized world coordinate, we set the translation as the height of camera,
        which is the position of the origin of the normalized coordinate system
        expressed in coordinates of the camera-centered coordinate system.
        :return:
        """
        Rc2n = np.eye(3, dtype=np.float64)
        Rc2n[1, 1] = math.cos(self.cam_pitch_rad)
        Rc2n[1, 2] = math.sin(self.cam_pitch_rad)
        Rc2n[2, 1] = -math.sin(self.cam_pitch_rad)
        Rc2n[2, 2] = math.cos(self.cam_pitch_rad)
        Rc2n = Rc2n.astype(np.float64)

        Tc2n = np.zeros((3, 1)).astype(np.float64)
        Tc2n[1] = -self.camHeight

        return Rc2n, Tc2n

    def undistort_point(self, points2d):
        """

        :param points2d:
        :return:
        """
        batch_size, num_kpt, feat_dim = points2d.shape
        points2d = np.reshape(points2d, (-1, 1, feat_dim))
        points2d = cv2.undistortPoints(points2d, self.K, self.dist_coeff, P=self.K)
        return np.reshape(points2d, (batch_size, num_kpt, feat_dim))

    def getRayInput(self, inputKeypoints):
        """
        :param inputKeypoints: matrix, (num, num_kpt,feat_dim)
        """
        inputKeypointsUndist = self.undistort_point(inputKeypoints)
        shape = inputKeypointsUndist.shape
        inputKeypointsUndistHomo = np.ones((shape[0], shape[1], shape[2] + 1), dtype=np.float64)
        inputKeypointsUndistHomo[..., :2] = inputKeypointsUndist
        pt_cam = inputKeypointsUndistHomo @ self.invK.T
        return pt_cam @ self.Rc2n.T