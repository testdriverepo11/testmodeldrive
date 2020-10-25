from tensorflow.keras.models import load_model
from parser import parser
import cv2
import os
import numpy as np
from camera import transform_img, eon_intrinsics
from model import medmodel_intrinsics
from lanes_image_space import transform_points
import json

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))


def frame_to_tensor(frame):
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    in_img1 = np.zeros((6, H // 2, W // 2), dtype=np.uint8)

    in_img1[0] = frame[0:H:2, 0::2] 
    in_img1[1] = frame[1:H:2, 0::2] 
    in_img1[2] = frame[0:H:2, 1::2]
    in_img1[3] = frame[1:H:2, 1::2]
    in_img1[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    in_img1[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
    return in_img1


def get_model_output(loaded_model,raw_frame,prev_frame2model,state=np.zeros((1, 512)),desire=np.zeros((1, 8))):
    parsed = outs = []
    frame_yuv420 = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YUV_I420)
    imgs_med_model = np.zeros((384, 512), dtype=np.uint8)
    imgs_med_model = transform_img(frame_yuv420, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,                                            output_size=(512, 256))
    frame_to_model = frame_to_tensor(np.array(imgs_med_model)).astype(np.float32) / 128.0 - 1.0
    if len(prev_frame2model):
        input_img_arr = np.vstack((prev_frame2model,frame_to_model))
        inputs = [input_img_arr[None], desire, state]
        outs = loaded_model.predict(inputs)
        parsed = parser(outs)

    prev_frame2model = frame_to_model;

    return [parsed,outs,prev_frame2model]


def model_thread():
    #pm = messaging.PubMaster(['model'])
    supercombo = load_model(BASEDIR+'/models/test.keras')
    x_left = x_right = x_path = np.linspace(0, 192, 192)
    cap = cv2.VideoCapture(BASEDIR+'/test.hevc')
    state = np.zeros((1, 512))
    desire = np.zeros((1, 8))
    prev_model_frame = []
    frame_counter=0
    #print(supercombo.summary())
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            model_output = get_model_output(supercombo, frame, prev_model_frame, state, desire)
            list_holder = [];
            i=0;
            parsed = model_output[0]
            raw_model_output = model_output[1]
            prev_model_frame = model_output[2];
            if len(raw_model_output):
                
                state = model_output[1][-1];
                for k in raw_model_output:
                    list_holder.insert(i,k.tolist()[0])
                    i=i+1
                print(list_holder)
             
            
def main():
 model_thread()

if __name__ == "__main__":
  main()
