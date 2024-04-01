from typing import NamedTuple, Tuple, Any
import cv2
import mediapipe as mp
from mediapipe import solutions
import math
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


class calculate_face_distance:

    def __init__(self, image) -> None:
        self.image = cv2.imread(image)


    def read_image(self) -> np.ndarray:
        converted_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return converted_image

    def setup_model(self, static_image_mode: bool = True, max_num_faces: int = 1,
                    min_detection_confidence: float = 0.5) -> NamedTuple:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces,
                                                    min_detection_confidence=min_detection_confidence)

        self.results = face_mesh.process(self.image)
        return self.results

    def get_image_dimensions(self) -> tuple[Any, Any]:
        # Get image dimensions
        self.image_rows, self.image_cols, _ = self.image.shape

        return self.image_rows, self.image_cols

    def find_two_landmarks(self, landmark_1: int, landmark_2: int) -> tuple:
        i = 0
        if self.setup_model().multi_face_landmarks:
            for face_landmarks in self.setup_model().multi_face_landmarks:

                for landmark in face_landmarks.landmark:
                    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                   self.get_image_dimensions()[1],
                                                                   self.get_image_dimensions()[0])

                    if landmark == face_landmarks.landmark[landmark_1]:
                        self.landmark_px1 = landmark_px

                    if landmark == face_landmarks.landmark[landmark_2]:
                        self.landmark_px11 = landmark_px

                    cv2.putText(self.image, f"{i}", landmark_px, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    i = i + 1


            return self.landmark_px1, self.landmark_px11

    def calculate_landmark_distance(self) -> float:

        if self.landmark_px1 and self.landmark_px11:
            distance_pixels = math.sqrt(
                (self.landmark_px11[0] - self.landmark_px1[0]) ** 2 + (
                        self.landmark_px11[1] - self.landmark_px1[1]) ** 2)

            # print("Distance between forehead landmarks in pixels:", distance_pixels)
            # Assuming your image has a known scale (pixels per centimeter), you can convert the distance to centimeters
            # For example, if 1 cm is represented by 10 pixels in your image

            pixels_per_cm = 10
            self.distance_cm = distance_pixels / pixels_per_cm
            # print("Distance between forehead landmarks in centimeters:", distance_cm)
            return self.distance_cm



    def show(self) -> cv2.imshow:


        cv2.circle(self.image, self.landmark_px1, 3, (0, 0, 255), 2)  # Yellow dot for forehead landmark 1
        cv2.circle(self.image, self.landmark_px11, 3, (0, 0, 255), 2)  # Yellow dot for forehead landmark 11

        cv2.line(self.image, self.landmark_px11, self.landmark_px1, (0, 200, 0), 3)  # Line between the two landmarks

        text_position = (
            (self.landmark_px1[0] + self.landmark_px11[0]) // 2, (self.landmark_px1[1] + self.landmark_px11[1]) // 2)
        cv2.putText(self.image, f"Distance: {self.distance_cm:.2f} cm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 5, cv2.LINE_AA)

        # Show the image with landmarks
        cv2.imshow("Face Landmarks", self.image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    face = calculate_face_distance("image.png")

    face.find_two_landmarks(10, 152)
    face.calculate_landmark_distance()
    face.show()

# 115.83