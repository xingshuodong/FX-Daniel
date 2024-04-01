import cv2
import mediapipe as mp
import math
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# Load the image
dframe = cv2.imread("image.png")

# Convert image to RGB (MediaPipe requires RGB input)
image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                            min_detection_confidence=0.5)

# Process the image
results = face_mesh.process(image_input)

# Get image dimensions
image_rows, image_cols, _ = dframe.shape
print(image_cols,image_rows)
# Initialize variables to store the coordinates of the two landmarks
landmark_px1 = None
landmark_px11 = None
i = 0
# Draw landmarks on the image
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        print(type(face_landmarks))
        for landmark in face_landmarks.landmark:
            # Convert normalized landmark coordinates to pixel coordinates
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            # print(landmark_px)

            cv2.putText(dframe, f"{i}", landmark_px, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            i = i + 1
            solutions.drawing_utils.draw_landmarks(
                image=dframe,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                # landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            #
            # solutions.drawing_utils.draw_landmarks(
            #     image=dframe,
            #     landmark_list=face_landmarks,
            #     connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
            #     # landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles
            #     .get_default_face_mesh_contours_style())

            # Find forehead landmark 1
            if landmark == face_landmarks.landmark[10]:
                landmark_px1 = landmark_px

            # Find forehead landmark 11
            if landmark == face_landmarks.landmark[152]:
                landmark_px11 = landmark_px
print(landmark_px11,landmark_px1)

# Calculate distance in pixels
if landmark_px1 and landmark_px11:
    distance_pixels = math.sqrt((landmark_px11[0] - landmark_px1[0]) ** 2 + (landmark_px11[1] - landmark_px1[1]) ** 2)
    print("Distance between forehead landmarks in pixels:", distance_pixels)

    # Assuming your image has a known scale (pixels per centimeter), you can convert the distance to centimeters
    # For example, if 1 cm is represented by 10 pixels in your image
    pixels_per_cm = 10
    distance_cm = distance_pixels / pixels_per_cm
    print("Distance between forehead landmarks in centimeters:", distance_cm)

    text_position = ((landmark_px1[0] + landmark_px11[0]) // 2, (landmark_px1[1] + landmark_px11[1]) // 2)
    cv2.putText(dframe, f"Distance: {distance_cm:.2f} cm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 5, cv2.LINE_AA)


# # Draw landmarks and lines on the image
if landmark_px1 and landmark_px11:
    cv2.circle(dframe, landmark_px1, 3, (0, 0, 255), 2)  # Yellow dot for forehead landmark 1
    cv2.circle(dframe, landmark_px11, 3, (0, 0, 255), 2)  # Yellow dot for forehead landmark 11
    cv2.line(dframe, landmark_px11, landmark_px1, (0, 200, 0), 3)  # Line between the two landmarks


# Show the image with landmarks
cv2.imshow("Face Landmarks", dframe)
cv2.waitKey(0)
cv2.destroyAllWindows()
