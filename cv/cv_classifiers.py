from fer import FER
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from datetime import datetime as dt


class fer_classifier():

  output_dir_name = "Output"
  cwd = os.getcwd()
  default_output_dir = os.path.join(os.getcwd(), output_dir_name)

  def __init__(self, mtcnn):
      
      self.mtcnn = mtcnn
      self.detector = FER(mtcnn=self.mtcnn)
      self.output_dir_name = "Output"
      self.cwd = os.getcwd()
      self.output_dir = os.path.join(self.cwd, self.output_dir_name)
      self.__setup_output_dir__()
  
  def __setup_output_dir__(self):     
          
     if(os.path.exists(self.output_dir_name) != True):
        os.mkdir(self.output_dir_name)
  
  def get_emotions(self, img_path, save_output, output_dir=default_output_dir):
     
    img = cv2.imread(img_path)
    result = self.detector.detect_emotions(img)    

    if(save_output):
      img_with_text = self.__draw_emotions__(img, result=result)
      self.__save_image__(img_with_text, output_dir)

    return result

  def get_top_emotion(self, img_path, save_output, output_dir=default_output_dir):
     
    img = cv2.imread(img_path)
    emotion_name, score = self.detector.top_emotion(img)

    if(save_output):
      self.__save_image_top_emotion__(emotion_name, score, img, output_dir)

    return emotion_name, score

  def __draw_emotions__(self, img, result):
     
    bounding_box = result[0]["box"]
    emotions = result[0]["emotions"]
    cv2.rectangle(img,(
      bounding_box[0], bounding_box[1]),(
      bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255), 2,)

    for index, (emotion_name, score) in enumerate(emotions.items()):

      color = (211, 211,211) if score < 0.01 else (128, 0, 128)
      emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))
    
      cv2.putText(img,emotion_score,
                  (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
                  cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)    
      
    return img

  def __save_image__(self, img, output_dir):

    os.chdir(output_dir)
    now = dt.now()
    date_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    file_name = f"{date_string}.jpg"
    cv2.imwrite(file_name, img)
    os.chdir(self.cwd)

  def __save_image_top_emotion__(self,emotion, emotion_score, img, output_dir):

    emotion_label = "{} - {}".format(emotion, emotion_score)
    cv2.putText(img, emotion_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)    
    os.chdir(output_dir)
    now = dt.now()
    date_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    file_name = f"{emotion}-{date_string}.jpg"
    cv2.imwrite(file_name, img)
    os.chdir(self.cwd)
  
