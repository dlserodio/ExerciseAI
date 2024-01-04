import cv2
import ScannerModule
from ScannerModule import BodyScanner
import time

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
main()