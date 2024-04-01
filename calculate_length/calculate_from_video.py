import cv2
import mediapipe as mp
import math
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe import solutions



class CalculateFaceDistance:

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence=0.5)
        self.results = None


    def find_landmarks(self, landmarks, landmark_indices) -> list:

        landmark_positions = []
        for indices in landmark_indices:
            landmark_set = []
            for index in indices:
                self.landmark_px = _normalized_to_pixel_coordinates(landmarks.landmark[index].x, landmarks.landmark[index].y, self.image_cols, self.image_rows)

                landmark_set.append(self.landmark_px)
            landmark_positions.append(landmark_set)

        return landmark_positions


    def calculate_landmark_distance(self, landmarks) -> list:



        cumulative_distances = []
        try:
            for landmark_set in landmarks:
                if landmark_set is not None and len(landmark_set) > 1:  # Add a check for None and minimum length
                    cumulative_distance = 0
                    for point in range(len(landmark_set) - 1):
                        distance_pixels = math.sqrt(
                            (landmark_set[point + 1][0] - landmark_set[point][0]) ** 2 +
                            (landmark_set[point + 1][1] - landmark_set[point][1]) ** 2
                        )
                        pixels_per_cm = 10  # Adjust this according to your image scale
                        distance_cm = distance_pixels / pixels_per_cm
                        cumulative_distance += distance_cm
                    cumulative_distances.append(cumulative_distance)
                else:
                    cumulative_distances.append(
                        0)  # Or any value to indicate no landmarks found or distance couldn't be calculated
            return cumulative_distances
        except:

            return cumulative_distances



    def process_frame(self, frame, landmark_indices_sets) -> None:
        converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(converted_image)
        i = 0

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:

                solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                    # landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                #
                landmark_positions = self.find_landmarks(face_landmarks, landmark_indices_sets)

                distances = self.calculate_landmark_distance(landmark_positions)
                for landmarks, distance in zip(landmark_positions, distances):
                    self.draw_landmark_set(frame, landmarks, distance)

    def draw_landmark_set(self, frame, landmarks, distance) -> None:

        for landmark in landmarks:

            cv2.circle(frame, landmark, 3, (0, 0, 255), 2)  # Yellow dot for each landmark
        text_position = ((landmarks[0][0] + landmarks[-1][0]) // 2, (landmarks[0][1] + landmarks[-1][1]) // 2)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        for i in range(len(landmarks) - 1):


            cv2.line(frame, landmarks[i], landmarks[i + 1], (0, 200, 0), 3)  # Line between the two landmarks

    def run(self, landmark_indices_sets) -> None:
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break



            self.image_rows, self.image_cols, _ = frame.shape
            self.process_frame(frame, landmark_indices_sets)
            cv2.imshow("Face Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_distance_calculator = CalculateFaceDistance()
    landmark_indices_sets = [

        [70, 63, 105, 66, 107],  # First set of landmark indices
        [336, 296, 334, 293, 300]  # Second set of landmark indices

        ]

    face_distance_calculator.run(landmark_indices_sets)
