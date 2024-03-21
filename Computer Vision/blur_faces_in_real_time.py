import cv2
import mediapipe as mp
import numpy as np
class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)

fl = FaceLandmarks()
cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  height, width, channels = frame.shape
  frame_copy = frame.copy()

  # 1. Face landmarks detection 
  landmarks = fl.get_facial_landmarks(frame)
  # print(landmarks)
  # pt = landmarks[0]

  # convexhull - the line joining all the outermost points of the face landmarks(face boundary)
  convexhull = cv2.convexHull(landmarks)
  cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)

  # 2. Face blurring
  mask = np.zeros((height, width), np.uint8) # creating a black image
  cv2.polylines(frame, [convexhull], True, 255, 3)
  cv2.fillConvexPoly(mask, convexhull, 255)

  # 3. Extract the face
  frame_copy = cv2.blur(frame_copy, (27,27))
  face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)
  #blurred_face = cv2.GaussianBlur(face_extracted, (27,27), 0)

  # 4. Extract the background
  background_mask = cv2.bitwise_not(mask)
  background = cv2.bitwise_and(frame, frame, mask = background_mask)

  # 5. Final result
  result = cv2.add(background, face_extracted)


  cv2.imshow('Frame', frame)
  cv2.imshow('Result', result)
  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()
