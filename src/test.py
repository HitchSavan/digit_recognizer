#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import os
import cv2
import numpy as np

def detect_symbols(im, symbols_height_threshold=10):
    script_folder = os.path.dirname(os.path.abspath(__file__))

    samples = np.loadtxt(os.path.join(script_folder, '..', 'data', 'generalsamples.data'), np.float32)
    responses = np.loadtxt(os.path.join(script_folder, '..', 'data', 'generalresponses.data'), np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    out = np.zeros(im.shape, np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    symbol_sequence_result = ''
    symbols_positions = {}

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > symbols_height_threshold:
            im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            string = str(int((results[0][0])))
            symbols_positions[(y, x)] = string
            cv2.putText(out, string, (x, y + h), 0, 0.5, (0, 255, 0))

    for pos in sorted(symbols_positions.keys()):
        symbol_sequence_result += symbols_positions[pos]

    return out, symbol_sequence_result

if __name__ == "__main__":
    script_folder = os.path.dirname(os.path.abspath(__file__))
    im = cv2.imread(os.path.join(script_folder, '..','data', 'test2.png'))

    out, symbol_sequence_result = detect_symbols(im)

    print(symbol_sequence_result)

    cv2.imshow('im', im)
    cv2.imshow('out', out)
    cv2.waitKey(0)