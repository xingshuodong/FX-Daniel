import sys
from typing import NamedTuple, Tuple, Any, List
import cv2
import mediapipe as mp
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
        self.results = face_mesh.process(self.read_image())
        return self.results
    def get_image_dimensions(self) -> tuple[Any, Any]:
        # Get image dimensions
        self.image_rows, self.image_cols, _ = self.image.shape

        return self.image_rows, self.image_cols

    def find_landmarks(self, landmark_1: int) -> tuple:

        if self.setup_model().multi_face_landmarks:
            for face_landmarks in self.setup_model().multi_face_landmarks:

                for landmark in face_landmarks.landmark:
                    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                   self.get_image_dimensions()[1],
                                                                   self.get_image_dimensions()[0])

                    if landmark == face_landmarks.landmark[landmark_1]:
                        self.landmark_pxl = landmark_px

            return self.landmark_pxl

    def get_landmark_index(self, landmark_index: np.array = [0]) -> np.ndarray:

        if len(landmark_index) < 2:
            print("Expected at least Two Landmarks")
            sys.exit()

        self.landmark_array = landmark_index

        return self.landmark_array

    def get_landmarks_pos(self) -> list[tuple]:

        self.landmark_positions = []

        for index in self.landmark_array:
            index = self.find_landmarks(index)
            self.landmark_positions.append(index)

        return self.landmark_positions

    def calculate_landmark_distance(self) -> float:
        landmarks = self.get_landmarks_pos()

        self.cumulative_distance = 0
        for point in range(len(landmarks) - 1):
            distance_pixels = math.sqrt(
                (landmarks[point + 1][0] - landmarks[point][0]) ** 2 +
                (landmarks[point + 1][1] - landmarks[point][1]) ** 2
            )

            pixels_per_cm = 10  # Adjust this according to your image scale
            distance_cm = distance_pixels / pixels_per_cm
            self.cumulative_distance += distance_cm
        return self.cumulative_distance

    # Draw landmarks and lines
    def get_draw_landmark_set(self, landmarks_sets: list[tuple]) -> None:

        print(self.calculate_landmark_distance())

        for landmarks in landmarks_sets:
            for landmark in landmarks:
                cv2.circle(self.image, landmark, 3, (0, 0, 255), 2)  # Yellow dot for each landmark
            # Draw lines between consecutive landmarks
            text_position = ((landmarks[0][0] + landmarks[-1][0]) // 2, (landmarks[0][1] + landmarks[-1][1]) // 2)
            cv2.putText(self.image, f"Distance: {self.calculate_landmark_distance():.2f} cm", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2, cv2.LINE_AA)
            for i in range(len(landmarks) - 1):
                cv2.line(self.image, landmarks[i], landmarks[i + 1], (0, 200, 0), 3)  # Line between the two landmarks

        return None

    def show(self) -> cv2.imshow:

        # # Show the image with landmarks and lines
        cv2.imshow("Face Landmarks", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face = calculate_face_distance("image.png")

    right_eye = face.get_landmark_index([70,63,105,66,107])
    right_eye = face.get_landmarks_pos()
    left_eye = face.get_landmark_index([336, 296,334,293,300])

    left_eye1 = face.get_landmarks_pos()
    face.get_draw_landmark_set([left_eye1,right_eye])


    face.show()
