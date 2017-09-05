# -*- coding: utf-8 -*-
from path import Path
import cv2
import numpy as np
from collections import OrderedDict


def normalize_image(img):
    clas = np.transpose(img.nonzero())
    img = center_image(img, clas)
    img = padding_img(img, board=2, resize=(28, 28))
    return img
    

def center_image(img0, clas):
    _, (tl, br) = order_points(clas)
    img = np.zeros_like(img0)
    for pos in clas:
        img[pos[0], pos[1]] = 255
    img = img[tl[0]: br[0] + 1, tl[1]: br[1] + 1]
    return img


def padding_img(img, board=0, resize=None):
    x, y = img.shape
    max_shape = max(img.shape)
    new_img = np.zeros((max_shape, max_shape), dtype='uint8')
    d = abs(x - y) // 2
    if x > y:
        new_img[:, d: y + d] = img
    elif x < y:
        new_img[d: x + d, ] = img
    else:
        new_img = img
    if board != 0 and resize is not None:
        new_size = (resize[0] - 2 * board, resize[1] - 2 * board)
    if resize is not None:
        resize_img = cv2.resize(new_img, new_size, cv2.INTER_AREA)
        board_img = np.zeros(resize, dtype='uint8')
        board_img[board: board + new_size[0], board: board + new_size[1]] = resize_img
        return board_img

    if board != 0:
        board_img = np.zeros((max_shape + 2 * board, max_shape + 2 * board), dtype='uint8')
        board_img[board: board + max_shape, board: board + max_shape] = new_img
        return board_img
    return new_img

                
def order_points(pts):
    pts = np.array(pts).reshape((-1, 2))
    rect1 = np.zeros((4, 2), dtype="int32")
    rect2 = np.zeros((2, 2), dtype="int32")
    s = pts.sum(axis=1)
    rect1[0] = pts[np.argmin(s)]
    rect1[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect1[3] = pts[np.argmin(diff)]
    rect1[1] = pts[np.argmax(diff)]

    min_x = np.argmin(pts[:, 0])
    min_y = np.argmin(pts[:, 1])

    max_x = np.argmax(pts[:, 0])
    max_y = np.argmax(pts[:, 1])

    rect2[0] = (pts[min_x][0], pts[min_y][1])
    rect2[1] = (pts[max_x][0], pts[max_y][1])
    return rect1, rect2
