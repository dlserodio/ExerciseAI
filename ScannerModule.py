import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

image_feed = cv2.VideoCapture(0)




class BodyScanner:
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.pose = mp_pose.Pose(mode, upBody, smooth, detectionCon, trackCon)
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def find_Pose(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return image
  
    def Position(self, frame, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmark_list
def main():
    image_feed = cv2.VideoCapture(0)
    mode = False
    onlyUp = True
    smooth = True
    detectCon = True
    trackCon = True
    
    body_scanner = BodyScanner(mode, onlyUp, smooth, detectCon, trackCon)

    while True:
        success, frame = image_feed.read()
        processed_frame = body_scanner.find_Pose(frame)
        landmark_list = body_scanner.Position(processed_frame, draw = False)

        if len(landmark_list) > 14: # Check if the index exists
            print(landmark_list[14])
            cv2.circle(frame, (landmark_list[14][1], landmark_list[14][2]), 10, (255, 0, 0), cv2.FILLED)
        else:
            print("The landmark at index 14 is not detected.")

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Live Video', processed_frame)

    cv2.destroyAllWindows()
    image_feed.release()

if __name__ == '__main__':
    main()