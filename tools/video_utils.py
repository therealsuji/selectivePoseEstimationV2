import os
import os.path as osp
import subprocess
import cv2
import glob
import numpy as np
import shutil


def video_to_images(vid_file, img_folder=None,):
    img_folder = 'output/tempfile/img'
    files = glob.glob(img_folder+"/*.png")
    for f in files:
        os.remove(f)
    os.makedirs(img_folder, exist_ok=True)
    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')
    # copies the first frame to separate folder for selection
    os.makedirs("output/tempfile/first_frame", exist_ok=True)
    shutil.copy2('output/tempfile/img/000001.png',
                 'output/tempfile/first_frame/000001.png')
    return img_folder


def images_to_video(img_folder, output_vid_file):
    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    return output_vid_file


def gen_masked_video(img_folder, trackers, track_id):

    image_file_names = sorted([
        osp.join(img_folder, x)
        for x in os.listdir(img_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])
    for frame, (img_fname, dets) in enumerate(zip(image_file_names, trackers)):
        image = cv2.imread(img_fname)
        for d in dets:
            d = d.astype(np.int32)
            track_id = int(track_id)
            if(track_id == d[4]):
                x = d[0]
                y = d[1]
                w = d[2] - d[0]
                h = d[3] - d[1]

                x = makeValueOverZero(x)
                y = makeValueOverZero(y)
                w = makeValueOverZero(w)
                h = makeValueOverZero(h)

                mask = np.zeros(image.shape, np.uint8)
                mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
                cv2.imwrite(img_fname, mask)


def makeValueOverZero(val):
    if val <= 0:
        return 0
    return val

def videoInfo(VideoName):
    cap = cv2.VideoCapture(VideoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, length

def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W>H:
            W_resize = max_length
            H_resize = int(H * max_length / W)
        else:
            H_resize = max_length
            W_resize = int(W * max_length / H)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize
    else:
        return frame, W, H