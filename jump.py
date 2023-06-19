from libs.camera import Camera
import libs.track.track as PersonTracker
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from libs.vis3dpose import plot_pose3d_m
from scipy.optimize import linear_sum_assignment
import math
from libs.one_euro_filter import OneEuroFilter
from libs.LandmarksSmoothingFilter import LandmarksSmoothingFilter
import time
import matplotlib
from packaging import version
import logging
import libs.vega_launcher as launcher
from libs.person_detector import PersonDetector
from libs.human_pose_detector import HumanPoseDetector
from libs.ray_3d import Ray3D
from ws import Websocket
import threading
import signal
from datetime import datetime
import dgmulticam
import arrow


ws = Websocket()

exitFlag = False
def exit_(signum,frame):
    print("exit_")
    global exitFlag
    exitFlag = True

def is_version_greater_than(pkg_version, required_version):
    pkg_version = version.parse(pkg_version)
    return pkg_version >= version.parse(required_version)


skeleton = ((0, 21), (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (9, 22), (6, 8), (8, 10), (10, 23), (5, 11), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (6, 12), (12, 14), (14, 16),
            (16, 18), (16, 20))

# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors]

def vis_keypoints(img, kps, kps_lines=skeleton, kp_thresh=0.3, alpha=1):

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def match(poseLList, poseRList):
    m = len(poseLList)
    n = len(poseRList)
    
    if m < 1 or n < 1:
        return None, None

    poseLArr = np.array(poseLList)
    poseRArr = np.array(poseRList)

    poseLArr_repeat = poseLArr[:, 0, :, :].reshape(m, 1, 24, 3).repeat(n, axis=1)
    poseRArr_repeat = poseRArr[:, 0, :, :].reshape(1, n, 24, 3).repeat(m, axis=0)
    mask = (poseLArr_repeat[..., 2] > 0.3) * (poseRArr_repeat[..., 2] > 0.3)
    y_diff = np.abs(poseLArr_repeat[..., 1] - poseRArr_repeat[..., 1])
    C = np.sum(y_diff * mask, axis=-1) / (np.sum(mask, axis=-1) + 1e-5)
    x_diff = (poseLArr_repeat[..., 0] - poseRArr_repeat[..., 0]) * mask
    rows, cols = linear_sum_assignment(C)
    assignDic = {}
    disparityList = []
    for row, col in zip(rows, cols):
        assignDic[row] = col
        disparityList.append(x_diff[row, col])
    return assignDic, np.array(disparityList)

def oks(pose1, pose2, scale):
    dSum = 0
    pNum = 0
    for i in range(24):
        d = np.linalg.norm(pose1[i, :2] - pose2[i, :2]) * (pose1[i, 2] > 0.3) * (pose2[i, 2] > 0.3)
        dSum += math.exp(-d / (2 * scale * 0.02))
        pNum += (pose1[i, 2] > 0.3) * (pose2[i, 2] > 0.3)

    return dSum / (pNum + 1e-8)

def oksMatch(detPoseList, poseInfoGT, detBboxs):
    trackIDListGt = []
    gtPoseList = []
    for key, info in poseInfoGT.items():
        trackIDListGt.append(key)
        gtPoseList.append(info["pose2d_cam"])

    m, n = len(detPoseList), len(gtPoseList)
    costMatrix = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        bbox = detBboxs[i]
        scaleBbox = math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        for j in range(n):
            costMatrix[i, j] = oks(detPoseList[i][0, :24], gtPoseList[j].T, scaleBbox)
    rows, cols = linear_sum_assignment(-costMatrix)

    detPoseTrackIDList = []
    for row, col in zip(rows, cols):
        if costMatrix[row, col] > 0.3:
            detPoseTrackIDList.append(trackIDListGt[col])
        else:
            detPoseTrackIDList.append(-1)

    return detPoseTrackIDList


class Pose3DInference:
    def __init__(self, pDet, PoseDet):
        self.peopleDetector = pDet
        self.poseDetector = PoseDet

        # 北京场地
        K = np.array([[1111.9407049196827, 0.0, 952.1068216246601], [0.0, 1106.2896979945215, 548.4763000360784], [0., 0., 1.]])
        dist = np.array([-0.33207135509744556, 0.1392927286472892, -7.059618025104669e-06, -8.412296356294607e-05, -0.03084351926645237])
        R = np.array([[0.9996221661928752,-0.025829445748187307,0.009400244050250606,], [-0.0034066506818656687,-0.4557749039790364,-0.890088552692389,], [0.027274889312233104,0.8897202237980439,-0.4556906887108446,]])
        T = np.array([-0.047914334102559915,  1.1812555918680145, 2.42872847646839]).reshape(3, 1)
        cam_football = Camera(K, dist,R=R, t=T)

        self.anatomyModel = Ray3D(cam_football, model_path="models/RayThreeD")

        self.body_tracker = PersonTracker.TrackPacket()
        self.body_tracker.set(1920, 1080)
        self.body_tracker.set_remove(False)
        self.idx_frame = 0
        self.colours = np.random.rand(32, 3) * 255  # used only for display
        self.colours = self.colours.astype(dtype=np.int32)

        self.detBboxFilter = {}
        self.pose2dFilter = {}
        self.pose3dFilter = {}

    def getTrackIDList(self, frameIndex):
        return [1]

    def inference(self, image):
        self.idx_frame += 1
        if image is None:
            return None, None

        detBBoxList = self.peopleDetector.detect(image)
        # track
        dets = list()
        dets_conf = list()
        det_result_features = list()
        for bbox_person in detBBoxList:
            det_result_features.append(None)
            dets.append([int(bbox_person[2]), int(bbox_person[3]), int(bbox_person[4]), int(bbox_person[5]), 2])
            dets_conf.append(1)
        dets = np.array(dets)
        self.body_tracker.predict()
        self.body_tracker.Update(dets.astype(dtype=np.int), det_result_features, dets_conf, self.idx_frame)
        current_det_result_ids = PersonTracker.get_current_det_result_track_id(dets.astype(dtype=np.int), self.body_tracker.alivePacket, self.idx_frame)

        image_draw = image.copy()
        keypoint_2d_dic_ori = {}
        keypoint_2d_dic_smooth = {}
        detPoseList = []
        for j in range(len(current_det_result_ids)):
            trackID = current_det_result_ids[j]
            if trackID not in  self.getTrackIDList(self.idx_frame):
                continue

            det = dets[j]
            ## smooth filter
            if trackID not in self.detBboxFilter.keys():
                self.detBboxFilter[trackID] = {'filter': OneEuroFilter(np.zeros(4), det[:4], min_cutoff=0.004, beta=0.7), 'startFrameIndex': self.idx_frame}
            else:
                preFrameIndex = self.detBboxFilter[trackID]['startFrameIndex']
                det[:4] = self.detBboxFilter[trackID]['filter'](self.idx_frame - preFrameIndex, det[:4])
            keypoint_2d = self.poseDetector.get_keypoints(image, [det[0], det[1], det[2], det[3]])
            if False:
                keypoint_2d = keypoint_2d[0].transpose()
                # smooth filter 2dpose 1€
                if trackID not in self.pose2dFilter.keys():
                    self.pose2dFilter[trackID] = {'filter': OneEuroFilter(np.zeros_like(keypoint_2d[:2, :]), keypoint_2d[:2, :], min_cutoff=0.004, beta=0.7, jitter_threshold=2), 'startFrameIndex': self.idx_frame}
                else:
                    preFrameIndex = self.pose2dFilter[trackID]['startFrameIndex']
                    keypoint_2d[:2, :] = self.pose2dFilter[trackID]['filter'](self.idx_frame - preFrameIndex, keypoint_2d[:2, :])
                image_draw = self.vis_keypoints(image_draw, keypoint_2d)
                keypoint_2d = keypoint_2d.transpose()
            else:
                keypoint_2d_smooth = keypoint_2d[0].copy()
                # smooth filter 2dpose LandmarksSmoothingFilter
                if trackID not in self.pose2dFilter.keys():
                    self.pose2dFilter[trackID] = {'filter': LandmarksSmoothingFilter(5, 10, keypoint_2d_smooth[:, :2].shape), 'startFrameIndex': self.idx_frame}
                else:
                    preFrameIndex = self.pose2dFilter[trackID]['startFrameIndex']
                    keypoint_2d_smooth[:, :2] = self.pose2dFilter[trackID]['filter'].apply(keypoint_2d_smooth[:, :2])
                # image_draw = self.vis_keypoints(image_draw, keypoint_2d_smooth.transpose())

            image_draw = self.vis_keypoints(image_draw, keypoint_2d[0].transpose())
            color = self.colours[trackID % 32, :]
            cl = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(image_draw, (dets[j][0], dets[j][1]), (dets[j][2], dets[j][3]), cl, 1)
            cv2.putText(image_draw, str(trackID), (dets[j][0] + 10, dets[j][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            # keypoint_2d_dic[trackID] = keypoint_2d[np.newaxis, :][:, :24, :]
            keypoint_2d_dic_smooth[trackID] = keypoint_2d_smooth[np.newaxis, :][:, :24, :]
            keypoint_2d_dic_ori[trackID] = keypoint_2d

        detPoseList = {}
        if len(keypoint_2d_dic_ori) < 1:
            return detPoseList, None, image_draw
        
        trackIdList, pose3d = self.anatomyModel.extract_keypoints3d(keypoint_2d_dic_smooth, self.idx_frame)
        pose3dAll = np.array(pose3d).reshape(-1,3).T
        #print(f'pose3dall: {pose3dAll}')
        
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        pose3d_ground = np.dot(R.T, pose3dAll)

        # pose3d_ground = pose3dAll
        pose3d_ground = pose3d_ground.reshape(3, -1, 24).transpose([1,0,2])
        pose3dAll_vis = np.concatenate([pose3d_ground,np.ones((pose3d_ground.shape[0],1,24))],axis=1)
        for trackid in trackIdList:
            detPoseList[trackID] = keypoint_2d_dic_ori[trackid][0].transpose()
        return detPoseList, pose3dAll_vis, image_draw

    def vis_keypoints(self, img, kps, kps_lines=skeleton, kp_thresh=0.4, alpha=1):
        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw the keypoints.
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def calcuateAngle(pose3d, start, media, end):
    # pose3d shape 4 x 24
    vec1 = pose3d[:3, start] - pose3d[:3, media]
    vec2 = pose3d[:3, end] - pose3d[:3, media]

    cosVal = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    sita = math.acos(cosVal) * 180 / math.pi

    return sita


class JumpJump:
    def __init__(self) -> None:
        self.L_angle = 180
        self.L_angle_min = 180
        self.L_shoulder_v = 0
        self.L_shoulder_last_y = None

        self.R_angle = 180
        self.R_angle_min = 180
        self.R_shoulder_v = 0
        self.R_shoulder_last_y = None

        self.squat_state = False
        self.lastJumpFrameIndex = -11
        self.angle_max = 180
        self.squat_threshold = 130

    def infer(self, pose3dList_Ray3D, frameIndex):
        jump_state = False
        direction_sita = -1

        if pose3dList_Ray3D is None:
            pose3dList_Ray3D = []
            self.L_shoulder_last_y = None
            self.R_hshoulder_last_y = None
            self.L_shoulder_v = 0
            self.R_shoulder_v = 0
            self.angle_max = 180
        else:
            self.L_angle = calcuateAngle(pose3dList_Ray3D[0], 13, 15, 17)
            self.L_angle_min = np.min([self.L_angle_min, self.L_angle])
            self.R_angle = calcuateAngle(pose3dList_Ray3D[0], 14, 16, 18)
            self.R_angle_min = np.min([self.R_angle_min, self.R_angle])
            self.angle_max = np.max([np.max([self.R_angle, self.L_angle]), self.angle_max])

            # if angle_max - (L_angle_min + R_angle_min) / 2 > 30:
            #     squat_threshold = (angle_max + (L_angle_min + R_angle_min) / 2) / 2

            if self.L_shoulder_last_y is not None:
                self.L_shoulder_v = pose3dList_Ray3D[0][2,7] - self.L_shoulder_last_y
                
            if self.R_shoulder_last_y is not None:
                self.R_shoulder_v = pose3dList_Ray3D[0][2,8] - self.R_shoulder_last_y

            self.L_shoulder_last_y = pose3dList_Ray3D[0][2, 7]
            self.R_shoulder_last_y = pose3dList_Ray3D[0][2, 8]

            # direction
            vec_shoulder = pose3dList_Ray3D[0][:,7] - pose3dList_Ray3D[0][:, 8]
            vec_shoulder_orth = np.array([vec_shoulder[1], -vec_shoulder[0], 0])
            direction_sita = math.acos(np.dot(vec_shoulder_orth, np.array([1, 0, 0])) / np.linalg.norm(vec_shoulder_orth)) * 180 / math.pi

        # if (self.L_angle + self.R_angle) / 2 < self.squat_threshold and self.squat_state == False and frameIndex - self.lastJumpFrameIndex > 10 and (self.L_shoulder_v + self.R_shoulder_v) / 2 < -0.015:
        if self.L_angle < self.squat_threshold and self.R_angle < self.squat_threshold and self.squat_state == False and frameIndex - self.lastJumpFrameIndex > 10 and (self.L_shoulder_v + self.R_shoulder_v) / 2 < -0.015:
            self.squat_state = True
            self.L_angle_min = 180
            self.R_angle_min = 180

        if self.squat_state and (self.L_angle + self.R_angle - self.L_angle_min - self.R_angle_min) / 2 > 10 and (self.L_angle + self.R_angle) / 2 > self.squat_threshold:
            self.squat_state = False
            self.lastJumpFrameIndex = frameIndex
            jump_state = True

        return [self.squat_state, jump_state, direction_sita, self.L_angle, self.L_shoulder_v, self.R_angle, self.R_shoulder_v, self.squat_threshold]


personDetector = None
humanPoseDetector = None
humanPose3DDetector = None
jumpJudge = None

def loadVega():
    launcher.load_vega()
    global personDetector,  humanPoseDetector, humanPose3DDetector, jumpJudge
    personDetector = PersonDetector("models/PersonDetAISports")
    humanPoseDetector = HumanPoseDetector("models/HumanPose")
    humanPose3DDetector = Pose3DInference(personDetector, humanPoseDetector)
    jumpJudge = JumpJump()


def run():
    # start websocket server
    server = threading.Thread(target=ws.newServer, args=('0.0.0.0', 9527))
    server.daemon = True
    server.start()
    
    # add dg camera and start
    dgmulticam.addCamera("192.168.100.110", 6660, True, 0)
    dgmulticam.setThreshold(1)
    dgmulticam.start()
    
    frameIndex = 0
    normal = 0

    while True:
        if exitFlag:
            break
        frameIndex += 1
        frames = dgmulticam.capture()
        if frameIndex % 3 != 0:
            continue
        #print(frames[0].shape)
        frameL = frames[0]["data"]
        timestamp = int(arrow.utcnow().float_timestamp * 1000)
        det2dPoseDic, pose3dList_Ray3D, outFrame = humanPose3DDetector.inference(frameL)
        squat_state, jump_state, direction_angle, L_angle, L_hip_v, R_angle, R_hip_v, squat_threshold = jumpJudge.infer(pose3dList_Ray3D, frameIndex)
        pushTime = int(arrow.utcnow().float_timestamp * 1000)
        print(f'infer cost: {pushTime-timestamp}')
        infer_result = []
        for k, v in det2dPoseDic.items():
            points = []
            for number in range(0, len(v[0])):
                x = v[0, number].astype(np.int32)
                y = v[1, number].astype(np.int32)
                points.append({'x': int(x), 'y': int(y)})
            infer_result.append({'frameId': frameIndex, "timestamp": timestamp, "pushTime": pushTime, "id": k, "squat": squat_state, "jump": jump_state, "directionAngle": direction_angle, "points": points})
        
        if len(infer_result) > 0:
            #infer_cost = pushTime - timestamp
            #print(f'infer cost {infer_cost} ms')
            ws.send_message(infer_result)

        if pose3dList_Ray3D is None:
            pose3dList_Ray3D = []

    dgmulticam.stop()
    dgmulticam.clear()
    ws.stop()
    time.sleep(1)
    server.join()


def debug():
    # start websocket server
    server = threading.Thread(target=ws.newServer, args=('0.0.0.0', 9527))
    server.daemon = True
    server.start()
    
    # write frame to video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("resources/out.mp4", fourcc, 30.0, (1920 + 540, 1080))
    
    # video data source
    capL = cv2.VideoCapture("resources/case1.mp4")
    assert capL.isOpened()
    retL, frameL = capL.read()
    #print(frameL.shape)
    frameIndex = 1
    while retL:
        if exitFlag:
            break
        if frameIndex % 10 == 0:
            print(frameIndex)
        timestamp = int(arrow.utcnow().float_timestamp * 1000)
        det2dPoseDic, pose3dList_Ray3D, outFrame = humanPose3DDetector.inference(frameL)
        squat_state, jump_state, direction_angle, L_angle, L_hip_v, R_angle, R_hip_v, squat_threshold = jumpJudge.infer(pose3dList_Ray3D, frameIndex)
        pushTime = int(arrow.utcnow().float_timestamp * 1000)
        #print(f'infer cost: {pushTime-timestamp}')
        # infer_result = []
        # for k, v in det2dPoseDic.items():
        #     points = []
        #     for number in range(0, len(v[0])):
        #         x = v[0, number].astype(np.int32)
        #         y = v[1, number].astype(np.int32)
        #         points.append({'x': int(x), 'y': int(y)})
        #     infer_result.append({'frameId': frameIndex, "timestamp": timestamp, "pushTime": pushTime, "id": k, "squat": squat_state, "jump": jump_state, "directionAngle": direction_angle, "points": points})
        # if len(infer_result) > 0:
        #     ws.send_message(infer_result)

        if pose3dList_Ray3D is None:
            pose3dList_Ray3D = []
        ## draw 3DPose
        fig = plt.figure(figsize=(5.4, 5.4))
        fig.add_subplot(projection='3d', adjustable='box')
        fig = plot_pose3d_m(pose3dList_Ray3D, fig, "Ray3D", 2, withDirection=True)
        plt.tight_layout()
        ax = fig.gca()
        ax.view_init(elev=67, azim=-90)
        fig.canvas.draw()
        image_3d3_front_ray3d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image_3d3_front_ray3d = image_3d3_front_ray3d.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        
        ax = fig.gca()
        ax.view_init(elev=8, azim=-90)
        fig.canvas.draw()
        image_3d3_side_ray3d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image_3d3_side_ray3d = image_3d3_side_ray3d.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)
        image_3d3_ray3d = np.concatenate((image_3d3_side_ray3d, image_3d3_front_ray3d), axis=0)
        img_concat_pose3d = np.concatenate((frameL, image_3d3_ray3d), axis=1)
        out.write(img_concat_pose3d)
        
        # read next frame
        retL, frameL = capL.read()
        frameIndex += 1
        
    print('debug over')
    ws.stop()
    time.sleep(1)
    server.join()
    out.release()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_)
    signal.signal(signal.SIGTERM, exit_)
    LOG_FORMAT = "%(levelname)s %(asctime)s | %(module)s:%(lineno)d  %(message)s"
    DATE_FORMAT = "%m-%d-%Y.%H:%M:%S.%p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    
    loadVega()
    # start jump judge
    # run()
    
    debug()