import os
import cv2
import glob
import copy
import math
import random
random.seed(0)
import argparse
import numpy as np
from numpy.linalg import eig, inv
import scipy.io as sio
from dotmap import DotMap
from PIL import ImageColor
import onnxruntime
from typing import List, Optional
from natsort import natsorted
from configparser import ConfigParser
from scrfd.scrfd_onnx import SCRFD

# FACE_LANDMARK_MODEL = 'slpt_decoder6_Nx3x256x256.onnx'
FACE_LANDMARK_MODEL = 'slpt_decoder12_Nx3x256x256.onnx'
ORIGINAL_IMAGE_SIZE_H = 450
ORIGINAL_IMAGE_SIZE_W = 450
WINDOWS_NAME = 'Inference'
LANDMARK_MODEL_INPUT_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0,
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def crop_v2(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )
    return dst_img, trans


def crop_img(img, bbox):
    x1, y1, x2, y2 = (bbox[:4]).astype(np.int32)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])
    scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) / 200.0
    img = img[..., ::-1]
    input, trans = crop_v2(img, center, scale * 1.15, (256, 256))
    return input.transpose(2,0,1), trans


def transform_pixel_v2(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ np.linalg.inv(trans[:,0:2].T)
    return pt


def main():
    DATASET = '300W_LP'
    TYPE = f'AFW'
    FOLDER = f'datasets/{DATASET}_croped/{TYPE}'
    image_files = glob.glob(f"{FOLDER}/*.jpg")
    mat_files = glob.glob(f"{FOLDER}/*.mat")

    # Load model
    ### Face Landmark - SLPT
    session_option_land = onnxruntime.SessionOptions()
    session_option_land.log_severity_level = 3
    face_landmark_sess = onnxruntime.InferenceSession(
        FACE_LANDMARK_MODEL,
        sess_options=session_option_land,
        providers=[
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    face_landmark_inputs = face_landmark_sess.get_inputs()

    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)

    ## Face Detection - SCRFD
    face_detection_model = 'scrfd_34g_1x3x480x480.onnx'
    detector = SCRFD(model_file=f'{os.getcwd()}/{face_detection_model}', nms_thresh=0.4)
    detector.prepare()

    for image_file, mat_file in zip(natsorted(image_files), natsorted(mat_files)):
        # Load image
        frame = cv2.imread(image_file)
        image = copy.deepcopy(frame)
        # Load .mat, .npy
        mat = sio.loadmat(mat_file)
        basename_without_ext = os.path.splitext(os.path.basename(image_file))[0]
        cp = np.load(f'{os.path.dirname(image_file)}/{basename_without_ext}_cp.npy')

        # Process
        heads = []
        # Calculate crop area based on mat file
        pt2d = mat['pt2d']
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        x_min = max(int(x_min), 0)
        y_min = max(int(y_min), 0)
        x_max = min(int(x_max), int(ORIGINAL_IMAGE_SIZE_W))
        y_max = min(int(x_max), int(ORIGINAL_IMAGE_SIZE_H))
        crop_start_x = 0
        crop_end_x = int(x_max-x_min)
        crop_start_y = 0
        crop_end_y = int(y_max-y_min)

        width = frame.shape[1]
        height = frame.shape[0]

        # Inference - Face Detection =================================== SCRFD
        heads, keypoints = detector.detect(
            img=frame,
            thresh=0.5,
            input_size=(frame.shape[1], frame.shape[0]),
        )

        if len(heads) > 0:
            heads = np.asarray(heads)
            head_images = []
            head_images_trans = []
            # If more than one person is in the image, only the person with the closest distance
            # from the center point of the angle of view and with the largest area is used.
            clean_heads = None
            if len(heads) > 1:
                prev_distance = 999999999999
                prev_area = 0
                for head in heads:
                    x_min = int(max(head[0], 0))
                    y_min = int(max(head[1], 0))
                    x_max = int(min(head[2], width))
                    y_max = int(min(head[3], height))
                    head[0] = x_min
                    head[1] = y_min
                    head[2] = x_max
                    head[3] = y_max
                    head_width = x_max - x_min
                    head_height = y_max - y_min

                    cp_x = x_min + (head_width // 2)
                    cp_y = y_min + (head_height // 2)

                    image_coord = np.array([cp[0], cp[1]])
                    bbox_coord = np.array([cp_y, cp_x])
                    distance = np.linalg.norm(bbox_coord - image_coord)

                    if prev_distance > distance:
                        prev_distance = distance
                        clean_heads = copy.deepcopy(head)

                    elif prev_distance == distance:
                        area = head_height * head_width
                        if prev_area < area:
                            prev_area = area
                            clean_heads = copy.deepcopy(head)

                heads = [clean_heads]

            # Facial landmark detection (only one face per image)
            for head in heads:
                x_min = int(max(head[0]-20, 0))
                y_min = int(max(head[1]-20, 0))
                x_max = int(min(head[2]+20, width))
                y_max = int(min(head[3]+20, height))
                head[0] = x_min
                head[1] = y_min
                head[2] = x_max
                head[3] = y_max

                # Pre-Process - Face Landmark ========================= SLPT 12
                # Generation of image batches for landmark detection
                alignment_input, trans = crop_img(frame, head)
                head_images.append(alignment_input)
                head_images_trans.append(trans)
                cv2.rectangle(
                    image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255,0,0),
                    2,
                    cv2.LINE_AA
                )

            head_images = np.asarray(head_images, dtype=np.float32)


        # Inference - Face Landmark
        """
        inputs[0].shape = [N, 3, 256, 256]
        outputs[0].shape = [N, 98, 2]
        """
        landmark_outputs = face_landmark_sess.run(
            None,
            {face_landmark_inputs[0].name: head_images},
        )
        landmarks = landmark_outputs[0]

        # Generate mask composite image
        """
        landmarks = [N, 98, 2] ... N x 98keypoints x XY
        """
        for landmark, head_image_trans in zip(landmarks, head_images_trans):
            for keypoints in landmark:
                landmark = transform_pixel_v2(
                    keypoints * LANDMARK_MODEL_INPUT_SIZE,
                    head_image_trans,
                    inverse=True,
                )
                landmark += 0.5
                landmark = landmark.astype(np.int32)
                cv2.circle(
                    image,
                    landmark,
                    radius=1,
                    color=(0,255,0),
                    thickness=-1,
                )
        processed_image =  image[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]

        cv2.imshow(WINDOWS_NAME, processed_image)
        key = cv2.waitKey(0)
        if key == 27: # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()