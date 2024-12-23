#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import os
import sys
import numpy as np
import cv2

def train_symbols(im, symbol_height_threshold=15):
    script_folder = os.path.dirname(os.path.abspath(__file__))
    im3 = im.copy()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0, 100), np.float32)
    responses = []
    keys = [i for i in range(48, 58)]

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > symbol_height_threshold:
            im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('press corresponding digit', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
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