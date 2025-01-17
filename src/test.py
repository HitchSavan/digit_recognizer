#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import os
import cv2
import numpy as np

def detect_symbols(im, symbols_height_threshold=10, symbols_width_threshold=10):
    script_folder = os.path.dirname(os.path.abspath(__file__))

    samples = np.loadtxt(os.path.join(script_folder, '..', 'data', 'generalsamples.data'), np.float32)
    responses = np.loadtxt(os.path.join(script_folder, '..', 'data', 'generalresponses.data'), np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    out = np.zeros(im.shape, np.uint8)
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 100, 255, 0)
    
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # # gray = cv2.bitwise_not(gray)
    # # blur = gray
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    # # _, thresh = cv2.threshold(blur, 150, 255, 0)


    # cv2.imshow('thresh', thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    symbol_sequence_result = ''
    symbols_positions = {}

    widths = []

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > symbols_height_threshold and w < symbols_width_threshold:
            im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            if dists > 1300000:
                # print(retval, dists)
                continue
            string = str(int((results[0][0])))
            if string != '-1':
                widths.append(w)
                symbols_positions[(x, y)] = string
                cv2.putText(out, string, (x, y + h), 0, 0.5, (0, 255, 0))

    avg_width = 0# sum(widths)/len(widths)
    skip = True
    prew_pos = [-1, -1]
    for pos in sorted(symbols_positions.keys()):
        if skip or pos[0] - prew_pos[0] > avg_width/2:
            symbol_sequence_result += symbols_positions[pos]
            skip = False
        prew_pos = pos

    return out, symbol_sequence_result

if __name__ == "__main__":
    script_folder = os.path.dirname(os.path.abspath(__file__))
    im = cv2.imread(os.path.join(script_folder, '..','data', 'test2.png'))

    out, symbol_sequence_result = detect_symbols(im, 15)

    print(symbol_sequence_result)

    cv2.imshow('im', im)
    cv2.imshow('out', out)
    cv2.waitKey(0)