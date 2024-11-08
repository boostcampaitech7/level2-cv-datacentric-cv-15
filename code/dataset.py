import os.path as osp
import math
import json
from PIL import Image

import time
import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon
import random
from numba import njit
from transform import *

@njit
def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices

@njit
def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

@njit
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


@njit
def is_cross_text_fast(start_loc, length, vertices):
    start_w, start_h = start_loc
    box_end_w, box_end_h = start_w + length, start_h + length
    for vertice in vertices:
        min_x, min_y = np.min(vertice[::2]), np.min(vertice[1::2])
        max_x, max_y = np.max(vertice[::2]), np.max(vertice[1::2])
        
        # 빠르게 겹치는지 확인
        if not (box_end_w < min_x or box_end_h < min_y or start_w > max_x or start_h > max_y):
            inter_area = max(0, min(box_end_w, max_x) - max(start_w, min_x)) * \
                         max(0, min(box_end_h, max_y) - max(start_h, min_y))
            p2_area = (max_x - min_x) * (max_y - min_y)
            if 0.01 <= inter_area / p2_area <= 0.99:
                return True
    return False

@njit
def random_starts(remain_w, remain_h, num_samples=10):
    # remain_w와 remain_h가 유효하지 않을 때 0으로 초기화된 int32 배열 반환
    if remain_w <= 0 or remain_h <= 0:
        return np.zeros(num_samples, dtype=np.int32), np.zeros(num_samples, dtype=np.int32)

    # remain_w와 remain_h가 유효할 때 randint로 int64 배열 생성
    starts_w = np.random.randint(0, remain_w, num_samples)
    starts_h = np.random.randint(0, remain_h, num_samples)

    # 수동으로 int32 배열로 변환
    return starts_w.astype(np.int32), starts_h.astype(np.int32)

@njit
def resize_vertices(vertices, ratio_w, ratio_h):
    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        # 리스트 인덱싱 대신 각각의 값을 개별적으로 접근
        for i in [0, 2, 4, 6]:
            new_vertices[:, i] = vertices[:, i] * ratio_w
        for i in [1, 3, 5, 7]:
            new_vertices[:, i] = vertices[:, i] * ratio_h
    return new_vertices

def crop_img(img, vertices, labels, length):
    h, w = img.height, img.width
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)

    ratio_w = img.width / w
    ratio_h = img.height / h
    new_vertices = resize_vertices(vertices, ratio_w, ratio_h)

    remain_h = max(0, img.height - length)
    remain_w = max(0, img.width - length)
    for _ in range(100):
        starts_w, starts_h = random_starts(remain_w, remain_h, num_samples=10)
        for start_w, start_h in zip(starts_w, starts_h):
            if not is_cross_text_fast([start_w, start_h], length, new_vertices[labels==1, :]):
                box = (start_w, start_h, start_w + length, start_h + length)
                region = img.crop(box)
                
                if new_vertices.size > 0:
                    new_vertices[:, [0, 2, 4, 6]] -= start_w
                    new_vertices[:, [1, 3, 5, 7]] -= start_h
                return region, new_vertices
            
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h

    # 기본값 반환 (좌표 못 찾았을 때)
    return region, new_vertices

@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    size = random.choice([2048, 2560, 3072])
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices

def resize_img_with_padding(img, vertices, target_size):
    h, w = img.height, img.width
    ratio = target_size / max(h, w)
    
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # 패딩 계산
    padding_top = pad_h // 2
    padding_left = pad_w // 2

    # 새로운 패딩 이미지 생성
    if random.random() < 0.7:
        # 밝은 색상 범위로 제한된 패딩 색상 생성
        padding_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img_padded = Image.new("RGB", (target_size, target_size), padding_color)
    else:
        padding_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_padded = Image.new("RGB", (target_size, target_size), padding_color)
    img_padded.paste(img_resized, (padding_left, padding_top))

    # 패딩을 고려한 새로운 vertices 계산
    new_vertices = vertices * ratio
    new_vertices[:, [0, 2, 4, 6]] += padding_left  # x 좌표에 왼쪽 패딩 추가
    new_vertices[:, [1, 3, 5, 7]] += padding_top   # y 좌표에 상단 패딩 추가

    return img_padded, new_vertices

def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):

        self._lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
            
        self.root_dir = root_dir
        self.split = split
        total_anno = dict(images=dict())
        
        # _lang_list에 있는 데이터셋들을 모두 불러오기
        for nation in self._lang_list:
            json_path = osp.join(root_dir, f'{nation}_receipt/ufo/{split}_corrected_v2.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                anno = json.load(f)
            for im in anno['images']:
                total_anno['images'][im] = anno['images'][im]
        
        self.anno = total_anno
        self.image_fnames = sorted(self.anno['images'].keys())

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def _infer_dir(self, fname):
        lang_indicator = fname.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        else:
            lang = 'sroie'
           
        return osp.join(self.root_dir, f'{lang}_receipt', 'img', self.split)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self._infer_dir(image_fname), image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts != 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        if random.random() < 0.7:
            image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        if random.random() < 0.5:
            image, vertices = crop_img(image, vertices, labels, self.crop_size)
        else:
            image, vertices = resize_img_with_padding(image, vertices, self.crop_size)


        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []

        if self.split == 'train':
            # texture
            funcs.append(
                A.OneOf([ 
                    A.Lambda(image=dirtydrum),  
                    A.Lambda(image=delaunay_pattern),
                    A.Lambda(image=dirtyscreen),
                    A.Lambda(image=dirthering),
                    A.Lambda(image=noise_texture),
            ], p=0.25)) 

            # words
            funcs.append(
                A.OneOf([  
                    A.Lambda(image=dotmatrix),  
                    A.Lambda(image=inkbleed),
                    A.Lambda(image=hollow),
                    A.Lambda(image=dilate),
                    A.Lambda(image=erode),
            ], p=0.2)) 

            # brightness
            funcs.append(
                A.OneOf([
                    A.ColorJitter(),
                    A.Lambda(image=lighting_gradient_gaussian),  
                    A.Lambda(image=lowlightness),
                    A.Lambda(image=shadowcast),
            ], p=0.5))

            funcs.append(A.Normalize())

            transform = A.Compose(funcs)
            image = transform(image=image)['image']
        elif self.split == 'val' and self.normalize:
            # Validation 시에는 Normalize만 적용
            transform = A.Normalize()
            image = transform(image=image)['image']

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask
