import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math
from packaging import version

def is_version_greater_than(pkg_version, required_version):
    pkg_version = version.parse(pkg_version)
    return pkg_version >= version.parse(required_version)

def plot_pose3d_one(pose, fig=None, pointType = 1, withDirection=False):
    """Plot the 3D pose showing the joint connections.
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10    
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle',  #16
                'Left tiptoe','Right tiptoe','Left heel','Right heel','Head top','Left hand','Right hand']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    if pointType == 1:
        _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [0, 5], [0, 6], [17, 15], [18, 16], [19, 15], [20, 16], [22, 9],
                    [23, 10]]
    else:
        _CONNECTION = [[0,1],[1,2],[1,23],[2,3],[2,4],[3,5],[4,6],[1,7],[1,8],[7,9],[9,11],[8,10],[10,12],
                    [0,13],[0,14],[13,15],[14,16],[15,17],[16,18],[17,19],[17,21],[18,22],[18,20]]

    def joint_color(j):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [5, 7, 9]:
            _c = 1
        if j in [6, 8, 10]:
            _c = 2
        if j in [11, 13, 15]:
            _c = 3
        if j in [12, 14, 16]:
            _c = 4
        # if j in range ( 17, 24 ):
        #     _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 4)
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(projection='3d', adjustable='box')
        
    ax = fig.gca()
    for c in _CONNECTION:
        if c[0] >= pose.shape[1] or c[1] >= pose.shape[1]:
            continue
        col = '#%02x%02x%02x' % joint_color(c[0])
        if pose[3, c[0]] > 0.3 and pose[3, c[1]] > 0.3:  # visable_flag[c[0]] and visable_flag[c[1]]:
            ax.plot([pose[0, c[0]], pose[0, c[1]]], [pose[1, c[0]], pose[1, c[1]]], [pose[2, c[0]], pose[2, c[1]]], c=col)

    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        if pose[3, j] > 0.3:
            if (j < 5 and j > 0) or j == 21:
                continue
            ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
            
    if withDirection:
        col = '#%02x%02x%02x' % joint_color(0)
        ax.plot([pose[0, 7], pose[0, 8]], [pose[1, 7], pose[1, 8]], [0, 0], c=col)
        vec_shoulder = pose[:, 7] - pose[:, 8]
        shoulder_center = (pose[:, 7] + pose[:, 8]) / 2.0
        vec_shoulder_orth = np.array([vec_shoulder[1], -vec_shoulder[0], 0])
        col = '#%02x%02x%02x' % (0, 0, 255)
        ax.plot([shoulder_center[0], shoulder_center[0] + vec_shoulder_orth[0]], [shoulder_center[1], shoulder_center[1] + vec_shoulder_orth[1]], [0, 0], c=col)

        sita = math.acos(np.dot(vec_shoulder_orth, np.array([1,0,0])) / np.linalg.norm(vec_shoulder_orth)) * 180 / math.pi
        ax.text(shoulder_center[0], shoulder_center[1] - 1.5, 0, "{:.02f}".format(sita))
            
    return fig

def plot_pose3d_m(poselist,fig, title="", pointType=1, withDirection=False):
    for pose in poselist:
        fig = plot_pose3d_one(pose, fig, pointType, withDirection)
        
    ax = fig.gca()
    
    ax.set_xlim(-5, 5)
    if withDirection:
        ax.set_ylim(0, 10)
    else:
        ax.set_ylim(-5, 5)
    ax.set_zlim(0,2)
    ax.set_aspect("equal")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=10, azim=-45)

    return fig
