import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

image_feed = cv2.VideoCapture(0)

while True:
    success, frame = image_feed.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)
    print(results.pose_landmarks)
    
    if cv2.waitKey(1) == 27:
        break
    elif results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, landmarks in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            print(id, landmarks)
            cx, cy = int(landmarks.x*w), int(landmarks.y*h)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)


    cv2.imshow('Live Video', frame)
cv2.destroyAllWindows()
image_feed.release()
