import numpy as np
import copy

from .tools import calculateFeatureDistanceMatrix2_total5_Matrix_8,calculateFeatureDistanceMatrix2_withKF
from .tools import iou, rect2xyah, xyah2rect,inboard, track_speed
from .tools import calculateDistance_with_KF, calculateDistance
from .kalman_filter import KalmanFilter

KalmanFilter = KalmanFilter()

distance_threshold_ = 10.0
kMaxFrameIntervalKeep = 60
initialState = None

feature_distance_threshold = 0.3
feature_distance_threshold_bigger = 0.5

person_border_top = 10
person_border_down = 10
person_border_left = 5
person_border_right = 5
person_bbox_minwidth = 70
person_bbox_minheigt = 180
person_border_aspectratio = 1.5
person_integrityscore_threshold = 0.9

class DetectObject(object):
    def __init__(self, frame_index, rect, score, feature, label):
        self.frame_index = frame_index
        self.rect = rect
        self.center = np.array([(rect[0] + rect[2]) / 2, (rect[3] + rect[1]) / 2])
        self.IntegrityScore = score
        self.feature = feature
        self.label = label
        self.removed = False

    def w(self):
        return self.rect[2] - self.rect[0]

    def h(self):
        return self.rect[3] - self.rect[1]

class FrameObjects(object):
    def __init__(self, frame_index, boxH, boxW):
        self.Frame_id = frame_index
        self.boxH = boxH
        self.boxW = boxW
        self.DetObj_lst = []

class TrackObject(object):
    def __init__(self, ID, mean, covariance):
        self.trk_id = ID
        self.MatchList = []
        self.deadFrames = 0
        self.state = "unconfirmed"
        self.mean = mean
        self.covariance = covariance

    def direction(self):
        if len(self.MatchList) < 3:
            return initialState
        else:
            return self.MatchList[-1].center - self.MatchList[-3].center

    def predict(self):
        self.mean, self.covariance = KalmanFilter.predict(self.mean, self.covariance)

    def Update(self, measurement):
        self.mean, self.covariance = KalmanFilter.update(self.mean, self.covariance, rect2xyah(measurement))

class DistanceUnit(object):
    def __init__(self, i, j, distance):
        self.i = i
        self.j = j
        self.distance = distance

class DistanceUnit2(object):
    def __init__(self, i, j, distance, distanceFeature):
        self.i = i
        self.j = j
        self.distance = distance
        self.distanceFeature = distanceFeature

def takeDistance(elem):  # elem:DistanceUnit
    return elem.distance

def takeDistance2(elem):  # elem:DistanceUnit
    return elem.distanceFeature

def get_det_result_lst(det_result, det_result_features, det_result_integ_score, current_frame_index, frame_w, frame_h, remove):
    FrameObjects_face = FrameObjects(current_frame_index, frame_h, frame_w)

    for i in range(len(det_result)):
        bndbox = det_result[i][:4]
        feature = det_result_features[i]
        integ_score = det_result_integ_score[i]
        label = 2 # to do
        obj = DetectObject(current_frame_index, bndbox, integ_score, feature, label)

        if remove:
            #图像不在下边缘的时候，完整度模型过滤
            if (bndbox[3] < frame_h - person_border_down) and (integ_score < 0.9):
                obj.removed = True
            #图像不在下边缘的时候，距离上边缘过滤
            if (bndbox[3] < frame_h - person_border_down) and (bndbox[1] < (person_border_top / 1080.0 * frame_h)):
                obj.removed = True
            ##图像不在下边缘的时候，最小高最小宽过滤
            if (bndbox[3] < frame_h - person_border_down) and ((bndbox[2] - bndbox[0]) < (person_bbox_minwidth / 1920.0 * frame_w) ) and ((bndbox[3] - bndbox[1]) < (person_bbox_minheigt / 1080.0 * frame_h) ):
                obj.removed = True
            #左右边界过滤
            if (bndbox[0] < person_border_left) and (bndbox[2] > (frame_w - person_border_right)):
                obj.removed = True
            #图像在下边缘的时候，高宽比过滤
            if (bndbox[3] > frame_h - person_border_down) and (((bndbox[3] - bndbox[1]) * 1.0 / (bndbox[2] - bndbox[0])) < person_border_aspectratio):
                obj.removed = True

        FrameObjects_face.DetObj_lst.append(obj)

    return FrameObjects_face.DetObj_lst


class TrackPacket(object):
    def __init__(self):
        self.alivePacket = []
        self.diePacket = []
        self.CurrentFrameIndex = 0
        self.Framelist = []
        self.ID_SUM = 0
        self.removed_person = []
        self.frame_w = 0
        self.frame_h = 0
        self.remove = False

    def set(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
    def set_remove(self, remove):
        self.remove = remove

    def trackSize(self):
        return len(self.alivePacket) + len(self.diePacket)

    def predict(self):
        for track in self.alivePacket:
            track.predict()

    def Update(self, det_result, det_result_features, det_result_integ_score, current_frame_index):
        FrameObjects = get_det_result_lst(det_result, det_result_features, det_result_integ_score, current_frame_index, self.frame_w, self.frame_h, self.remove)

        tmpPacket = [i for i in self.alivePacket]
        self.removed_person = []
        self.alivePacket = []
        trackobject_matched = list(range(len(tmpPacket)))
        detectobject_matched = list(range(len(FrameObjects)))

        #cal feature distance and location distance
        Match_distance = []

        if len(tmpPacket) != 0 and len(FrameObjects) != 0:
            RectDistance = calculateFeatureDistanceMatrix2_withKF(tmpPacket, FrameObjects)

        for i in trackobject_matched:
            for j in detectobject_matched:
                Match_distance.append(DistanceUnit2(i, j,  RectDistance[i, j],1))

        track_match_removed = [False] * len(tmpPacket)

        # sort by location dis
        Match_distance.sort(key=takeDistance)

        for distance_unit in Match_distance:
            if (distance_unit.i in trackobject_matched) and (distance_unit.j in detectobject_matched):

                distance_threshold = distance_threshold_ * 1.

                if distance_unit.distance < distance_threshold:

                    if FrameObjects[distance_unit.j].removed == True and track_match_removed[distance_unit.i]==False:
                        track_match_removed[distance_unit.i] = True
                        tmpPacket[distance_unit.i].Update(FrameObjects[distance_unit.j].rect)

                    if track_match_removed[distance_unit.i] == True:
                        continue

                    trackobject_matched.remove(distance_unit.i)
                    detectobject_matched.remove(distance_unit.j)
                    tmpPacket[distance_unit.i].MatchList.append(FrameObjects[distance_unit.j])

                    tmpPacket[distance_unit.i].deadFrames = 0
                    tmpPacket[distance_unit.i].Update(FrameObjects[distance_unit.j].rect)

                    self.alivePacket.append(tmpPacket[distance_unit.i])

        # if track not matched
        for i in trackobject_matched:
            frame_interval = current_frame_index - tmpPacket[i].MatchList[-1].frame_index
            if frame_interval > kMaxFrameIntervalKeep:
                self.diePacket.append(tmpPacket[i])
            elif inboard(tmpPacket[i].MatchList[-1].rect, self.frame_w, self.frame_h) and (
                    frame_interval > kMaxFrameIntervalKeep / 2):
                self.diePacket.append(tmpPacket[i])
            elif inboard(tmpPacket[i].MatchList[-1].rect, self.frame_w, self.frame_h) and (
                    track_speed(tmpPacket[i]) > 2) and (frame_interval > 15):
                self.diePacket.append(tmpPacket[i])
            else:
                self.alivePacket.append(tmpPacket[i])

        # if det not matched
        for j in detectobject_matched:
            obj = FrameObjects[j]
            if obj.removed == True:
                mean, covariance = KalmanFilter.initiate(rect2xyah(obj.rect))
                TrObj = TrackObject(-1, mean, covariance)
                TrObj.MatchList.append(obj)
                self.removed_person.append(TrObj)
                continue
            self.ID_SUM += 1
            mean, covariance = KalmanFilter.initiate(rect2xyah(obj.rect))
            TrObj = TrackObject(self.ID_SUM, mean, covariance)
            TrObj.MatchList.append(obj)
            self.alivePacket.append(TrObj)


def get_current_det_result_track(det_result, alivePacket, removed_person, frame_index):
    current_det_result_obj = []
    track_all = alivePacket + removed_person
    for bbox in det_result:
        for trackObj in track_all:
            if trackObj.MatchList[-1].frame_index == frame_index:
                bndbox = trackObj.MatchList[-1].rect
                if bbox[:4] == bndbox:
                    current_det_result_obj.append(trackObj)
                    break

    return current_det_result_obj


def get_current_det_result_track_id(det_result, alivePacket, frame_index):
    current_det_result_ids = []
    for bbox in det_result:
        for trackObj in alivePacket:
            if trackObj.MatchList[-1].frame_index == frame_index:
                bndbox = trackObj.MatchList[-1].rect
                if all(bbox[:4] == bndbox):
                    current_det_result_ids.append(trackObj.trk_id)
                    break
        else:
            current_det_result_ids.append(-1)
    return current_det_result_ids
