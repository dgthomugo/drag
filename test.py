import logging
import libs.vega_launcher as launcher
from libs.person_detector import PersonDetector
from libs.human_pose_detector import HumanPoseDetector
import cv2


def test_person_detect(image):
    # use vega to decode image
    img = cv2.imread(image)
    if img is None:
        print("Error: can't load image ")
        exit(-1)
    detector = PersonDetector("models/PersonDetAISports")
    result = detector.detect(img)
    logging.info(f'detected person succeed: {result}')
    return result
    

def test_human_pose(image):
    boxes = test_person_detect(image)
    box = boxes[0]
    img = cv2.imread(image)
    if img is None:
        print("Error: can't load image ")
        exit(-1)
    detector = HumanPoseDetector("models/HumanPose")
    result = detector.get_keypoints(img, [box[2], box[3], box[4], box[5]])
    logging.info(f'human pose: {result}')
    

if __name__ == "__main__":
    LOG_FORMAT = "%(levelname)s %(asctime)s | %(module)s:%(lineno)d  %(message)s"
    DATE_FORMAT = "%m-%d-%Y.%H:%M:%S.%p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    launcher.load_vega()
    image_uri = "resources/person1.jpg"
    test_human_pose(image_uri)