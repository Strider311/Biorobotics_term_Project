import cv.webcam_component as webcam_component
import cv.cv_classifiers as cv_classifiers
from cv.cv_tester import dataset_tester 
import cv2

jaffe_path = "datasets\\FER\\jaffe"
cv_emotion_test = dataset_tester(jaffe_path, "jaffe")

cv_emotion_test.run()