import numpy as np
import torch
def Condition_one(pts, length = 5, threshold=0.0002):
    hip_joint = torch.mean(pts[:, 11:13, 1], axis=1)
    velocity = torch.abs(torch.sub(hip_joint[-1], hip_joint[-5], alpha = 1) / length)
    return velocity >= threshold

def Condition_two(pts, threshold = 45):
    s_bar = torch.mean(pts[-1,15:17,:2], axis=0)
    y = pts[-1,0,1] - s_bar[1]
    x = abs(pts[-1,0,0] - s_bar[0])
    theta =  np.rad2deg(np.arctan(y/x))
    return theta < threshold 

def Condition_three(bbox, threshold = 0.5):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    p = width / height
    return p >= threshold