from cv import cv_classifiers
import os
from datetime import datetime as dt
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class fer_tester():

    def __init__(self, dataset_path, dataset_name):

        self.dataset_name = dataset_name
        self.cwd = os.getcwd()
        self.full_path = os.path.join(self.cwd, dataset_path)
        self.emotions_path_dic = {}
        self.output_dir_name = "Test_Results"
        self.emotions = os.listdir(self.full_path)
        self.output_path = os.path.join(self.cwd, self.output_dir_name)

        self.emotion_map = {}
        index = 0
        for emotion in self.emotions:
            self.emotion_map[emotion] = index
            index += 1

        now = dt.now()
        date_string = now.strftime("%d-%m-%Y_%H-%M-%S")    
        self.dataset_dir = f"{self.dataset_name}-{date_string}" 
        self.result_path = os.path.join(self.output_path, self.dataset_dir)

        self.__fill_path_dic__()
        self.__setup_output_dir__()
        self.emotion_classifier = cv_classifiers.fer_classifier(mtcnn=True)

    def __fill_path_dic__(self):

        for emotion in self.emotions:
            self.emotions_path_dic[emotion] = os.path.join(self.full_path, emotion)

    def __setup_output_dir__(self):

        if(os.path.exists(self.output_path) != True):
            os.mkdir(self.output_path)
        
        os.chdir(self.output_dir_name)
        os.mkdir(self.dataset_dir)                
        os.chdir(self.dataset_dir)

        for emotion in self.emotions:
            os.mkdir(emotion)
        
        os.chdir(self.cwd)

    def __setup_output_csv(self):
        
        column_names = ['file_name','expected_emotion','detected_emotion','score', 'expected_int', 'detected_int']
        columns = [y for x in [column_names, self.emotions] for y in x]
        self.result_df = pd.DataFrame(columns=column_names)
        
    def run(self):
        
        self.__setup_output_csv()

        fail_count = {}
        emotion_img_length_dic = {}
        success_rate_dic = {}
        zeros = [0]*(len(self.emotions))
        df_index = 0

        for emotion in self.emotions:

            emotion_path = os.path.join(self.full_path, emotion)
            images = os.listdir(emotion_path)
            output_path = os.path.join(self.output_dir_name,self.dataset_dir, emotion)
            fail_count[emotion] = 0
            emotion_img_length_dic[emotion] = len(images)

            for image in images:
                
                image_path = os.path.join(emotion_path, image)
                detected_emotion, score = self.emotion_classifier.get_top_emotion(img_path=image_path, save_output=True, output_dir=output_path)
                print(f"{image} expected: {emotion}, result: {detected_emotion} - {score}")
                if (detected_emotion != emotion):
                    fail_count[emotion] = fail_count[emotion]+1

                    
                data = [image, emotion, detected_emotion, score, self.emotion_map[emotion], self.emotion_map[detected_emotion]]                
                # updated_entry = data + zeros
                self.result_df.loc[df_index] = data
                # self.result_df.loc[df_index, detected_emotion] = 1         

                df_index += 1
                self.result_df = self.result_df.sort_index()

            success_score = (emotion_img_length_dic[emotion] - fail_count[emotion])/(emotion_img_length_dic[emotion]) * 100
            success_rate_dic[emotion] = "{:.2f}".format(success_score)
                
        os.chdir(self.result_path)
        output_name = f"{self.dataset_name}-results.csv"
       
        self.result_df.to_csv(output_name, header=True, index=False, encoding='utf-8')

        print("\n---------------------------------")
        os.chdir(self.cwd)
        print(f"\nSuccess score %: \n{success_rate_dic}\n")

        self.__confusion_matrix__()
        
    def __confusion_matrix__(self):

        print("\n---------------------------------")
        print(f"\nEmotions mapping:\n{self.emotion_map}\n")
        actual = pd.Series(self.result_df["expected_int"], name="Actual")
        predicted = pd.Series(self.result_df["detected_int"], name="Predicted")
        
        cf_matrix = pd.crosstab(actual, predicted, margins=True)
        print("\n---------------------------------\n")
        print(cf_matrix)
        print("\n---------------------------------\n")

        plt.matshow(cf_matrix)
        


