import csv
import sys
import cv2
import urllib.request as urlreq
import os
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams

def overlayMask(pic, mask_img_name, mask_csv):

    image = pic
    result = image.copy()
    result = result.astype(np.float32) / 255.0
    
    x, y, width, depth = 50, 200, 950, 500

    image_gray = image

    haarcascade = 'haarcascade_frontalface_alt2.xml'

    detector = cv2.CascadeClassifier()

    faces = detector.detectMultiScale(image_gray)
    print(faces)
    for face in faces:
        (x,y,w,d) = face
        cv2.imwrite("test.png", cv2.rectangle(image,(x,y),(x+w, y+d),(255, 255, 255), 2))
    
    LBFmodel = "lbfmodel.yaml"

    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    _, landmarks = landmark_detector.fit(image_gray, faces)

    landmarks = np.reshape(np.array(landmarks), (-1,2))
    dst_pts = np.array(
        [
            landmarks[1],
            landmarks[2],
            landmarks[3],
            landmarks[4],
            landmarks[5],
            landmarks[6],
            landmarks[7],
            landmarks[8],
            landmarks[9],
            landmarks[10],
            landmarks[11],
            landmarks[12],
            landmarks[13],
            landmarks[14],
            landmarks[15],
            landmarks[29],
        ],
        dtype="float32"
    )
    mask = mask_img_name
    mask_annotation = mask_csv

    with open(mask_annotation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        src_pts = []
        for i, row in enumerate(csv_reader):
            try:
                src_pts.append(np.array([float(row[1]), float(row[2])]))
            except ValueError:
                continue
    src_pts = np.array(src_pts, dtype="float32")

    if(landmarks > 0).all():
        mask_img = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        mask_img = mask_img.astype(np.float32)/255

        M, _ = cv2.findHomography(src_pts, dst_pts)
        transformed_mask = cv2.warpPerspective(
            mask_img,
            M,
            (result.shape[1], result.shape[0]),
            None,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT,
        )
        #print(M.shape)
        #print(transformed_mask.shape)
        alpha_mask = transformed_mask[:,:,3]
        alpha_image = 1.0 - alpha_mask
        #print(result.shape)
        #print(transformed_mask.shape)
        #for c in range(0,3):
        #    result[:,:,c] -= 3*transformed_mask[:,:,c]
        
        #relu = np.zeros((result.shape[0], result.shape[1])) 
        #for c in range(0,3):
        #    result[:,:,c] = np.maximum(result[:,:,c], relu)
        for c in range(0,3):
            result[:,:,c] = (
               alpha_mask * transformed_mask[:,:,c]
               +  alpha_image * result[:,:,c]
            )
        
    return result*255
if __name__ == "__main__":
    dataset = "fer2013.csv"
    mask_img_name = "./masks/mask.png"
    mask_csv = "./masks/mask.csv"
    for i in range(1,5):
        img = cv2.imread("test{}.jpeg".format(i))
        print(img.shape)
        cv2.imwrite("mask_{}.png".format(i), overlayMask(img, mask_img_name, mask_csv))
    '''with open(dataset) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            str_list = list(row[1].split(" "))
            img = np.array([float(v) for v in str_list])
            img = np.reshape(img, (48,48))
            cv2.imwrite("img.png", img)

            mask_img = overlayMask(img)
            imwrite("test.png", mask_img)
            sys.exti(0)'''
