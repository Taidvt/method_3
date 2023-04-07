import numpy as np

def Condition_one(pts, length, threshold=0.09):
    hip_joint = np.mean(pts[:, 7:9, 1], axis=1)
    velocity = np.abs(np.diff(hip_joint)[-1] / length)
    return velocity >= threshold

def Condition_two(pts, threshold = 45):
    threshold = 45
    s_bar = np.mean(pts[-1,11:13,:2], axis=0)
    y = pts[-1,0,1] - s_bar[1]
    x = abs(pts[-1,0,0] - s_bar[0])
    theta =  np.rad2deg(np.arctan(y/x))
    return theta < threshold 

def Condition_three(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    p = width / height
    return p >= 0.5