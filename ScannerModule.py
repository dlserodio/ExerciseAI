import cv2
import mediapipe as mp
import math

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
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.landmark_list
    
    def AngleFinder(self, processed_frame, p1, p2, p3, draw  =True):
        x1, y1  = self.landmark_list[p1][1:]
        x2, y2  = self.landmark_list[p2][1:]
        x3, y3  = self.landmark_list[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        print(angle)
        if draw:
            cv2.circle(processed_frame, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(processed_frame, (x2, y2), 10, (0, 255, 0), 2)
            cv2.circle(processed_frame, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(processed_frame, str(int(angle)), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

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
        processed_frame = body_scanner.find_Pose(frame, False)
        landmark_list = body_scanner.Position(processed_frame, draw = False)

        if len(landmark_list) != 0:
            body_scanner.AngleFinder(processed_frame,24, 26, 28)
        else:
            print("The landmarks not detected.")

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Live Video', processed_frame)

    cv2.destroyAllWindows()
    image_feed.release()

if __name__ == '__main__':
    main()