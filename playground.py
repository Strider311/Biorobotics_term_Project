from cv import webcamComponent



# img_path = "media\Saber-photo.JPG"cl
# img = cv2.imread(img_path)
# detector = FER(mtcnn=True)
# result = detector.detect_emotions(img)
# emotion_name, score = detector.top_emotion(img)


# bounding_box = result[0]["box"]
# emotions = result[0]["emotions"]
# cv2.rectangle(img,(
#   bounding_box[0], bounding_box[1]),(
#   bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#               (0, 155, 255), 2,)

# for index, (emotion_name, score) in enumerate(emotions.items()):
#    color = (211, 211,211) if score < 0.01 else (128, 0, 128)
#    emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))
 
#    cv2.putText(img,emotion_score,
#                (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
#                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
 
# #Save the result in new image file
# cv2.imwrite("media\\emotion.jpg", img)
# # Read image file using matplotlib's image module
# result_image = mpimg.imread('media\\emotion.jpg')
# imgplot = plt.imshow(result_image)
# # Display Output Image
# plt.show()


webcam = webcamComponent.WebCamComponent(0, True)
webcam.run()