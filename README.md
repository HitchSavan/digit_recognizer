# Simple Digit Recognition OCR in OpenCV Python

=============================================

This code is originally based on Abid Rahman K's answer in [Simple Digit Recognition OCR in OpenCV-Python](http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python/9620295).

Forked and modified from [eyanq/sdr](https://github.com/eyanq/sdr).

## Progress

Run `train.py`, press corresponding key to label the number surrounded with blue rectangle. The labeled result will seems like

![Train Result](/data/train_result.png)

Run `test.py`, the program will load the test image and automatically recognize digits using `KNearNeighbour` Algorithm. The results will seem like

Labeled Test Image

![Labeled Test Image](/data/in.png)

Recognized Digits

![Recognized Digits](/data/out.png)

## Development Environment Info

```python
>>> sys.version
'3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)]'

>>> numpy.__version__
'1.24.4'

>>> cv2.__version__
'4.10.0'
```
