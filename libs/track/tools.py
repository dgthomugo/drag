import numpy as np

iou_weight = 5.15
frame_weight = 2.0
pos_weight = 3.295
scale_weight = 1.635
feature_weight = 4.0
type_weight = 1.0

def iou(b1, b2):#[xmin, ymin, xmax, ymax]
    iou_val = 0.0

    b1_w = b1[2] - b1[0]
    b1_h = b1[3] - b1[1]
    b2_w = b2[2] - b2[0]
    b2_h = b2[3] - b2[1]

    x1 = np.max([b1[0], b2[0]])
    y1 = np.max([b1[1], b2[1]])
    x2 = np.min([b1[2], b2[2]])
    y2 = np.min([b1[3], b2[3]])
    w = np.max([0, x2 - x1])
    h = np.max([0, y2 - y1])
    if w != 0 and h != 0:
        iou_val = float(w * h) / (b1_w * b1_h + b2_w * b2_h - w * h)

    return iou_val

def inboard(rect, frame_w, frame_h, kBoardToDrop=30):
    if (rect[0] < kBoardToDrop) or (rect[1] < kBoardToDrop) or (rect[2] > frame_w - kBoardToDrop) or (
            rect[3] > frame_h - kBoardToDrop):
        return True
    else:
        return False

def track_speed(track):
    if len(track.MatchList) > 2:
        direct = track.direction()
        frame_interval = track.MatchList[-1].frame_index - track.MatchList[-3].frame_index
        velocity = direct / frame_interval
        speed = np.sqrt(np.sum((np.square(velocity))))
        return speed
    else:
        return 0

def rect2xyah(rect):
    xyah = np.array(rect).copy()
    xyah = xyah.astype(float)
    xyah[2:] = xyah[2:] - xyah[:2]
    xyah[:2] = xyah[:2] + xyah[2:] / 2
    xyah[2] = xyah[2] / xyah[3]
    return list(xyah)

def xyah2rect(xyah):
    rect = np.array(xyah).copy()
    rect = rect.astype(float)
    rect[2] = rect[2] * rect[3]
    rect[:2] = rect[:2] - rect[2:]/2
    rect[2:] = rect[:2] + rect[2:]
    return list(rect)

def calculateDistance(object1, object2):
    if isinstance(object1, DetectObject) and isinstance(object2, DetectObject):
        frame_interval = abs(object2.frame_index - object1.frame_index)
        max_unit = max(object1.h(), object2.h(), object1.w(), object2.w())
        mean_unit = (object1.h() + object2.h() + object1.w() + object2.w()) / 4

        iou_distance = 1.0 - iou(object1.rect, object2.rect)
        frame_distance = (frame_interval - 1) * 0.015
        pos_distance = np.sqrt(np.sum(np.square(object1.center - object2.center))) / max_unit
        scale_distance = np.sqrt(
            np.sum(np.square(np.array([object2.h(), object2.w()]) - np.array([object1.h(), object1.w()])))) / mean_unit
        feature_distance = 0
        type_distance = 0
        distance = iou_weight * iou_distance + frame_weight * frame_distance + pos_weight * pos_distance + scale_weight * scale_distance + feature_weight * feature_distance + type_weight * type_distance

        return distance

def calculateDistance_with_KF(track1, object2, feature_distance):
    object1 = track1.MatchList[-1]
    mean = track1.mean[:4]
    rect = xyah2rect(mean)

    frame_interval = abs(object2.frame_index - object1.frame_index)
    max_unit = max(mean[3], object2.h(), mean[2] * mean[3], object2.w())
    mean_unit = (mean[3] + object2.h() + mean[2]*mean[3] + object2.w())/4
    iou_distance = 1.0 - iou(rect, object2.rect)
    frame_distance = (frame_interval - 1) * 0.015
    pos_distance = np.sqrt(np.sum(np.square(mean[:2] - object2.center))) / max_unit
    scale_distance = np.sqrt(np.sum(np.square(np.array([object2.h(), object2.w()]) - np.array([mean[3], mean[2]*mean[3]])))) / mean_unit
    feature_distance = 0
    type_distance = 0

    distance = iou_weight * iou_distance + frame_weight * frame_distance + pos_weight * pos_distance + scale_weight * scale_distance + feature_weight * feature_distance + type_weight * type_distance
    
    return distance 

def calculateFeatureDistanceMatrix2_withKF(TrackObjects, FrameObjects):
    m = len(TrackObjects) + 1
    n = len(FrameObjects) + 1
    matrix = np.ones((m,n),dtype=np.float) * 1000
    for i in range(len(TrackObjects)):
        for j in range(len(FrameObjects)):
            matrix[i,j] = calculateDistance_with_KF(TrackObjects[i],FrameObjects[j],0)

    return matrix

def cal_sim_matrix2(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)


def calculateFeatureDistanceMatrix2_total5_Matrix_8(TrackObjects, FrameObjects):
    TrackObjects_feature = []
    FrameObjects_feature = []
    num_of_fea_of_eachTrack_for_calDis = [0]
    min_dis = np.ones((len(TrackObjects), len(FrameObjects)))
    for i in range(len(TrackObjects)):
        if len(TrackObjects[i].MatchList) < 10:
            for j in range(len(TrackObjects[i].MatchList)):
                TrackObjects_feature.append(TrackObjects[i].MatchList[j].feature)
            num_of_fea_of_eachTrack_for_calDis.append(len(TrackObjects[i].MatchList) )            
        # else:
            # for j in range(len(TrackObjects[i].MatchList) ):
            #     TrackObjects_feature.append(TrackObjects[i].MatchList[j].feature)
            # num_of_fea_of_eachTrack_for_calDis.append(len(TrackObjects[i].MatchList) )
        else:
            index = [-10, -9, -8, -7, -6,-5, -4 , - 3]
            for j in index:
                TrackObjects_feature.append(TrackObjects[i].MatchList[j].feature)
            num_of_fea_of_eachTrack_for_calDis.append(8)
    for i in range(len(FrameObjects)):
        FrameObjects_feature.append(FrameObjects[i].feature)
    sim = cal_sim_matrix2(TrackObjects_feature, FrameObjects_feature)
    dis = 1 - sim
    j = 0
    index = np.cumsum(num_of_fea_of_eachTrack_for_calDis)
    for i in range(len(index) - 1):
        min_dis[j] = np.min(dis[index[i]:index[i+1]],axis=0)
        j += 1
    return min_dis 


