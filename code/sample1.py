import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import ImageEnhance as eh
import tensorflow as tf
import os

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


i=0
cap = cv2.VideoCapture('./videos/v3.mp4')
while True:
    ret, frame = cap.read()
    img = frame
    cla = cv2.createCLAHE(4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    #eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)

    gc_image = adjust_gamma(img,0.8)


    output2 = tf.image.adjust_saturation(gc_image, 1.9)
    #output3 = output2.eval(session=tf.compat.v1.Session())

    Hori = np.concatenate((img, output2), axis=1)
    cv2.imshow('HORIZONTAL', Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(i)
    if i==1000:
        break
    j=0
    while j<20:
        ret, frame = cap.read()
        j+=1
    i+=50