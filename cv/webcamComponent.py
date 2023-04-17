from fer import FER
import cv2
import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class WebCamComponent():
    def __init__(self, device, mtcnn):
        self.mtcnn = mtcnn
        self.device = device
        pass

    def draw_scores(self,
        frame: np.ndarray,
        emotions: dict,
        bounding_box: dict,
        lang: str = "en",
        size_multiplier: int = 1,
    ) -> np.ndarray:
    
        GRAY = (211, 211, 211)
        GREEN = (0, 255, 0)
        x, y, w, h = bounding_box

        for idx, (emotion, score) in enumerate(emotions.items()):
            color = GRAY if score < 0.01 else GREEN

            if lang != "en":
                emotion = self.emotions_dict[emotion][lang]

            emotion_score = "{}: {}".format(
                emotion, "{:.2f}".format(score) if score >= 0.01 else ""
            )
            cv2.putText(
                frame,
                emotion_score,
                (
                    x,
                    y + h + (15 * size_multiplier) + idx * (15 * size_multiplier),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * size_multiplier,
                color,
                1 * size_multiplier,
                cv2.LINE_AA,
            )
        return frame


    def draw_annotations(self,
        frame: np.ndarray,
        faces: list,
        boxes=True,
        scores=True,
        color: tuple = (0, 155, 255),
        lang: str = "en",
        size_multiplier: int = 1,
    ) -> np.ndarray:
        """Draws boxes around detected faces. Faces is a list of dicts with `box` and `emotions`."""
        if not len(faces):
            return frame

        for face in faces:
            x, y, w, h = face["box"]
            emotions = face["emotions"]

            if boxes:
                cv2.rectangle(
                    frame,
                    (x, y, w, h),
                    color,
                    2,
                )

            if scores:
                frame = self.draw_scores(
                    frame, emotions, (x, y, w, h), lang, size_multiplier)
        return frame


    def run(self):
        detector = FER(mtcnn=self.mtcnn)
        cap = cv2.VideoCapture(self.device)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.flip(frame, 1)
            emotions = detector.detect_emotions(frame)
            frame = self.draw_annotations(frame, emotions)

            # Display the resulting frame
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
