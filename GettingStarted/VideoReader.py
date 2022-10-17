import cv2 as cv
import numpy as np

def main():
    capture = cv.VideoCapture('valorant.mp4')
    video_tensor = []
    while True:
        ret, frame = capture.read()
        if ret == True:
            video_tensor.append(frame.tolist())
        else:
            break
    video_tensor = np.array(video_tensor)
    print(video_tensor.shape)

if __name__ == "__main__":
    main()
