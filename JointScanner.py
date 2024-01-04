import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class BodyScanner:
    def __init__(self, mode, onlyUp, smooth, detectCon, trackCon):
        self.pose = mp_pose.Pose(mode, onlyUp, smooth, detectCon, trackCon)

    def process_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, landmarks in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(landmarks.x * w), int(landmarks.y * h)
                cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return image

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
        processed_frame = body_scanner.process_image(frame)

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Live Video', processed_frame)

    cv2.destroyAllWindows()
    image_feed.release()

if __name__ == '__main__':
    main()