import cv.webcam_component as webcam_component
import cv.cv_classifiers as cv_classifiers
from cv.cv_tester import fer_tester 
import cv2

jaffe_path = "datasets\\FER\\jaffe"
cv_emotion_test = fer_tester(jaffe_path, "jaffe")

# ck_path = "datasets\\FER\\CK"
# cv_emotion_test = fer_tester(ck_path, "CK")

cv_emotion_test.run()