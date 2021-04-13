import cv2
import numpy as np

#-----------
from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import Flask,render_template,Response, request,redirect, url_for



classifier = load_model('./model/emotion_little_vgg_2.h5')#Pretrained Model usage
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"

from os import listdir
from os.path import isfile, join
import os
import cv2

m_hash = 'fbe63257a054c1c5466cfd7bf14646d6'
emotion_list = {0: 'Angry',
                1: 'Fear',
                2: 'Happy',
                3: 'Neutral',
                4: 'Sad',
                5: 'Surprise'}  # Emotion Classes


def display_emotion(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


# model parameters defined
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

# weights
if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=m_hash, cache_dir=Path(sys.argv[0]).resolve().parent)
# Loading the PRETRAINED MODELS
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

detector = dlib.get_frontal_face_detector()
#end------------


class VideoCamera(object):#VIDEO CAMERA FUNCTION OF POLYGON SHAPE DETECTION MODULE


    def __init__(self):
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,frame=self.video.read()

        preprocessed_faces_emo = []

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        preprocessed_faces_emo = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)

            # Gender and Age Prediction
            results = model.predict(np.array(faces))
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # Emotion Display Loop
            emo_labels = []
            for i, d in enumerate(detected):
                preds = classifier.predict(preprocessed_faces_emo[i])[0]
                emo_labels.append(emotion_list[preds.argmax()])
                # TEXT NOTIFIER: Real time customer emotional status in text
                if preds.argmax() == 0:
                    print("Customer is ANGRY!")  # Diffuse the situation
                elif preds.argmax() == 1:
                    print("Customer is in FEAR!")  # Something bad is happening
                elif preds.argmax() == 2:
                    print("Customer is HAPPY!")  # happy customers come back for more
                elif preds.argmax() == 3:
                    print("Customer is NEUTRAL.")  # customer is just here for something
                elif preds.argmax() == 4:
                    print("Customer is SAD...")  # try to make a deal that customer likes
                elif preds.argmax() == 5:
                    print("Customer is SURPRISED!")  # this can be good or bad depending what the surprise is
                else:  # no face detected
                    print("Nothing is registered!")  # no face detect, pass
            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}, {}".format(int(predicted_ages[i]),
                                            "F" if predicted_genders[i][0] > 0.4 else "M", emo_labels[i])
                display_emotion(frame, (d.left(), d.top()), label)




        ret,jpeg=cv2.imencode('.jpg',frame)

        return jpeg.tobytes()