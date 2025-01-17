#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import os
import sys
import numpy as np
import cv2

def train_symbols(im, symbol_height_threshold=15):
    script_folder = os.path.dirname(os.path.abspath(__file__))

    samples = np.empty((0, 100), np.float32)
    responses = []
    keys = [i for i in range(48, 58)]

    frames_folder_path = os.path.join(script_folder, '..', 'data', 'frames')

    frames = next(os.walk(frames_folder_path), (None, None, []))[2]  # [] if no file
    
    
    for frame_file in frames:
        frame = cv2.imread(os.path.join(frames_folder_path, frame_file))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(gray, 100, 255, 0)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # gray = cv2.bitwise_not(gray)
        # # blur = gray
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
        # # _, thresh = cv2.threshold(blur, 150, 255, 0)

        cv2.imshow('test_threshold', thresh)
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        key = 0

        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)

            if h > symbol_height_threshold:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow('press corresponding digit', frame)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    break
                    # sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)
                else:
                    responses.append(-1)
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        if key == 27:  # (escape to quit)
            break

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    print('training complete')

    samples = np.float32(samples)
    responses = np.float32(responses)

    cv2.imwrite(os.path.join(script_folder, '..', 'data', 'train_result.png'), im)
    np.savetxt(os.path.join(script_folder, '..', 'data', 'generalsamples.data'), samples)
    np.savetxt(os.path.join(script_folder, '..', 'data', 'generalresponses.data'), responses)

if __name__ == "__main__":
    script_folder = os.path.dirname(os.path.abspath(__file__))
    im = cv2.imread(os.path.join(script_folder, '..', 'data', 'train.png'))

    train_symbols(im)