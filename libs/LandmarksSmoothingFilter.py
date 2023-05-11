import numpy as np
import time
from collections import deque, namedtuple

# Filtering
class LowPassFilter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.initialized = False
    def apply(self, value):
        # Note that value can be a scalar or a numpy array
        if self.initialized:
            v = self.alpha * value + (1.0 - self.alpha) * self.stored_value
        else:
            v = value
            self.initialized = True
        self.stored_value = v
        return v
    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

# RelativeVelocityFilter : https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/relative_velocity_filter.cc
# This filter keeps track (on a window of specified size) of
# value changes over time, which as result gives us velocity of how value
# changes over time. With higher velocity it weights new values higher.
# Use @window_size and @velocity_scale to tweak this filter.
# - higher @window_size adds to lag and to stability
# - lower @velocity_scale adds to lag and to stability
WindowElement = namedtuple('WindowElement', ['distance', 'duration'])
class RelativeVelocityFilter:
    def __init__(self, window_size, velocity_scale, shape=1):
        self.window_size = window_size
        self.velocity_scale = velocity_scale
        self.last_value = np.zeros(shape)
        self.last_value_scale = np.ones(shape)
        self.last_timestamp = -1
        self.window = deque()
        self.lpf = LowPassFilter()

    def apply(self, value_scale, value, timestamp=None):
        # Applies filter to the value.
        # timestamp - timestamp associated with the value (for instance,
        #             timestamp of the frame where you got value from)
        # value_scale - value scale (for instance, if your value is a distance
        #               detected on a frame, it can look same on different
        #               devices but have quite different absolute values due
        #               to different resolution, you should come up with an
        #               appropriate parameter for your particular use case)
        # value - value to filter
        if timestamp is None:
            timestamp = time.perf_counter()
        if self.last_timestamp == -1:
            alpha = 1.0
        else:
            distance = value * value_scale - self.last_value * self.last_value_scale
            duration = timestamp - self.last_timestamp
            cumul_distance = distance.copy()
            cumul_duration = duration
            # Define max cumulative duration assuming
            # 30 frames per second is a good frame rate, so assuming 30 values
            # per second or 1 / 30 of a second is a good duration per window element
            max_cumul_duration = (1 + len(self.window)) * 1/30
            for el in self.window:
                if cumul_duration + el.duration > max_cumul_duration:
                    break
                cumul_distance += el.distance
                cumul_duration += el.duration
            velocity = cumul_distance / cumul_duration
            alpha = 1 - 1 / (1 + self.velocity_scale * np.abs(velocity))
            self.window.append(WindowElement(distance, duration))
            if len(self.window) > self.window_size:
                self.window.popleft()

        self.last_value = value
        self.last_value_scale = value_scale
        self.last_timestamp = timestamp

        return self.lpf.apply_with_alpha(value, alpha)

def get_object_scale(landmarks):
    # Estimate object scale to use its inverse value as velocity scale for
    # RelativeVelocityFilter. If value will be too small (less than
    # `options_.min_allowed_object_scale`) smoothing will be disabled and
    # landmarks will be returned as is.
    # Object scale is calculated as average between bounding box width and height
    # with sides parallel to axis.
    # landmarks : numpy array of shape nb_landmarks x 3
    lm_min = np.min(landmarks[:2], axis=1) # min x + min y
    lm_max = np.max(landmarks[:2], axis=1) # max x + max y
    return np.mean(lm_max - lm_min) # average of object width and object height

class LandmarksSmoothingFilter:
    def __init__(self, window_size = 5, velocity_scale = 10, shape=1):
        # 'shape' is shape of landmarks (ex: (33, 3))
        self.window_size = window_size
        self.velocity_scale = velocity_scale
        self.shape = shape
        self.init = True

    def apply(self, landmarks):
        # landmarks = numpy array of shape nb_landmarks x 3 (3 for x, y, z)
        # Here landmarks are absolute landmarks (pixel locations)
        if self.init: # Init or reset
            self.filters = RelativeVelocityFilter(self.window_size, self.velocity_scale, self.shape)
            self.init = False
            out_landmarks = landmarks
        else:
            value_scale = 1 / get_object_scale(landmarks)
            out_landmarks = self.filters.apply(value_scale, landmarks)
        return out_landmarks

    def reset(self):
        if not self.init: self.init = True
