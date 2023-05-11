import cv2
import numpy as np
import copy


class AdjustDisparity:
    def __init__(self) -> None:
        self.PATCH_SIZE = 20
        self.REGION_SIZE = 30

    def take_patch(self, im, y, x, size):
        size_w, size_h = size
        canvas = np.zeros((size_h * 2, size_w * 2, 3), im.dtype)
        ybegin = y - size_h
        yoffset = 0
        if ybegin < 0:
            yoffset = -ybegin
            ybegin = 0

        yend = y + size_h
        yoffset2 = size_h * 2
        if yend > im.shape[0]:
            yoffset2 = size_h * 2 + (im.shape[0] - yend)
            yend = im.shape[0]

        xbegin = x - size_w
        xoffset = 0
        if xbegin < 0:
            xoffset = -xbegin
            xbegin = 0

        xend = x + size_w
        xoffset2 = size_w * 2
        if xend > im.shape[1]:
            xoffset2 = size_w * 2 + (im.shape[1] - xend)
            xend = im.shape[1]
        canvas[yoffset:yoffset2, xoffset:xoffset2, :] = im[ybegin:yend, xbegin:xend, :]

        return canvas

    def adjust(self, imgL, imgR, matchDic, detPoseList, disparity):
        detPoseListOrder = []
        for idx, idx_ in matchDic.items():
            detPoseListOrder.append(detPoseList[idx])
        if  len(detPoseListOrder) < 1:
            return disparity

        detPoseArr = np.array(detPoseListOrder).reshape(-1, 3)
        disparityArr = disparity.flatten()
        assert len(detPoseArr) == len(disparityArr)

        disparity_ = copy.deepcopy(disparityArr)
        for i in range(len(detPoseArr)):
            if detPoseArr[i, 2] < 0.3 or disparityArr[i] == 0:
                continue
            img_h,img_w,_ = imgL.shape
            if int(detPoseArr[i, 1]) < 0 or int(detPoseArr[i, 1]) >= img_h:
                continue
            if int(detPoseArr[i, 0]) < 0 or int(detPoseArr[i, 0]) >= img_w or int(detPoseArr[i, 0] - disparityArr[i]) < 0 or int(detPoseArr[i, 0] - disparityArr[i]) >= img_w:
                continue
            
            target = self.take_patch(imgL, int(detPoseArr[i, 1]), int(detPoseArr[i, 0]), (self.PATCH_SIZE, self.PATCH_SIZE))
            region = self.take_patch(imgR, int(detPoseArr[i, 1]), int(detPoseArr[i, 0] - disparityArr[i]), (self.REGION_SIZE, self.PATCH_SIZE))

            res = cv2.matchTemplate(region, target, cv2.TM_CCOEFF_NORMED)
            _, _, _, top_left = cv2.minMaxLoc(res)
            matchIndex = top_left[0]

            if res[0, (matchIndex + 1) % 21] - res[0, (matchIndex - 1) % 21] > 0:
                matchIndex += 0.5
                
            disparity_[i] = int(detPoseArr[i, 0]) - int(detPoseArr[i, 0] - disparityArr[i]) + 10 - matchIndex

        return disparity_.reshape(-1, 24)
