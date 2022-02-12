import yaml
import cv2
import numpy as np

def load_yaml(file_path):
    return yaml.load(open(file_path, 'r'), Loader=yaml.SafeLoader)

## Linemod
def trans2StandardBBOX(bbox:list):
    pass

def transRT2Matrix4X4(R: list, t: list):
    R_np = np.array(R).reshape(3, 3)
    t_np = np.array(t).reshape(3, 1)
    matrix4X3 = np.concatenate((R_np, t_np), axis=1)
    return list(matrix4X3.flatten())

def calculateKeyPoint2D(points3D, intrinsic, RT):
    RT_np = np.array(RT).reshape(3, 4)

    points3D_np = np.array(points3D).reshape(8, 3)
    pad = np.ones((8, 1))
    points3D_np = np.concatenate((points3D_np, pad), axis=1).T

    tmp = np.matmul(np.matmul(intrinsic, RT_np), points3D_np)
    tmp[0, :] = tmp[0, :] / tmp[2, :]
    tmp[1, :] = tmp[1, :] / tmp[2, :]
    res = tmp[:2, :].T.flatten()
    return list(res)

## draw bbox
def draw2DBBOX(image, bbox):
    bbox = [int(item) for item in bbox]
    point1 = (bbox[0], bbox[1])
    point2 = (bbox[0], bbox[1]+bbox[3])
    point3 = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    point4 = (bbox[0]+bbox[2], bbox[1])

    color = (0, 255, 255)
    thickness = 2
    cv2.line(image, point1, point2, color, thickness)
    cv2.line(image, point2, point3, color, thickness)
    cv2.line(image, point3, point4, color, thickness)
    cv2.line(image, point4, point1, color, thickness)

    return image

def draw3DBBOX(image, bbox):
    bbox = [int(item) for item in bbox]
    point1 = tuple(bbox[:2])
    point2 = tuple(bbox[2:4])
    point3 = tuple(bbox[4:6])
    point4 = tuple(bbox[6:8])
    point5 = tuple(bbox[8:10])
    point6 = tuple(bbox[10:12])
    point7 = tuple(bbox[12:14])
    point8 = tuple(bbox[14:])

    color = (0, 255, 255)
    thickness = 1

    cv2.line(image, point1, point2, color, thickness)
    cv2.line(image, point2, point3, color, thickness)
    cv2.line(image, point3, point4, color, thickness)
    cv2.line(image, point4, point1, color, thickness)

    cv2.line(image, point5, point6, color, thickness)
    cv2.line(image, point6, point7, color, thickness)
    cv2.line(image, point7, point8, color, thickness)
    cv2.line(image, point8, point5, color, thickness)

    cv2.line(image, point1, point5, color, thickness)
    cv2.line(image, point2, point6, color, thickness)
    cv2.line(image, point3, point7, color, thickness)
    cv2.line(image, point4, point8, color, thickness)

    return image


