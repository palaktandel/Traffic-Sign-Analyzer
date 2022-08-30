#code for videocapture with camera
import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if ret == False:
            continue
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()