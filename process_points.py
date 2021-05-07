import os
import cv2
from tqdm import tqdm
import numpy as np
import time
from hrnet.pose_estimation.video import getTwoModel, generate_2d_keypoints
from tools.utils import videopose_model_load as Model3Dload
from tools.utils import draw_3Dimg, draw_2Dimg,generate_3d_keypoints
from tools.video_utils import videoInfo, resize_img
bboxModel, poseModel = getTwoModel()
model3D = Model3Dload()


def process_video(VideoName):
    cap, cap_length = videoInfo(VideoName)
    kpt2Ds = []
    print(cap_length)
    kpt3d = []
    for i in tqdm(range(cap_length)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            t0 = time.time()
            joint2D = generate_2d_keypoints(bboxModel, poseModel, frame)  
            print('HrNet comsume {:0.3f} s'.format(time.time() - t0))
        except Exception as e:
            print(e)
            continue

        if i == 0:
            for _ in range(30):
                kpt2Ds.append(joint2D)
        elif i < 30:
            kpt2Ds.append(joint2D)
            kpt2Ds.pop(0)
        else:
            kpt2Ds.append(joint2D)

        joint3D = generate_3d_keypoints(model3D, np.array(kpt2Ds))
        joint3D_item = joint3D[-1] #(17, 3)
        kpt3d.append(joint3D_item)
        # draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)
    np.save('outputfile', kpt3d)
    


