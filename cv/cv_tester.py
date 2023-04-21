from cv import cv_classifiers
import os
from datetime import datetime as dt


class fer_tester():

    def __init__(self, dataset_path, dataset_name):

        self.dataset_name = dataset_name
        self.cwd = os.getcwd()
        self.full_path = os.path.join(self.cwd, dataset_path)
        self.emotions_path_dic = {}
        self.output_dir_name = "Test_Results"
        self.emotions = os.listdir(self.full_path)
        self.output_path = os.path.join(self.cwd, self.output_dir_name)

        now = dt.now()
        date_string = now.strftime("%d-%m-%Y_%H-%M-%S")    
        self.dataset_dir = f"{self.dataset_name}-{date_string}" 
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

    def run(self):
        
        fail_count = {}
        emotion_img_length_dic = {}
        success_rate_dic = {}
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
            
            success_score = (emotion_img_length_dic[emotion] - fail_count[emotion])/(emotion_img_length_dic[emotion]) * 100
            success_rate_dic[emotion] = "{:.2f}".format(success_score)
        
        print(f"Success rate: \n{success_rate_dic} ")
        

        


