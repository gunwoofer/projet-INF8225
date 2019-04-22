import numpy as np
import cv2
import argparse
from main import get_frame


#Parsing des arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="./output/output.avi",
	help="path to input gray color video")
ap.add_argument("-o", "--output", type=str, default="./output/output_color.avi",
	help="path to output video")
ap.add_argument("-m", "--model", type=str, default="1",
	help="path to output video")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["input"])

ret, frame = cap.read()
print('ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])


FPS= 30.0
FrameSize=(frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

frames = []
i = 0
while(cap.isOpened()):
    print(i)
    ret, frame = cap.read()

    # check for successfulness of cap.read()
    if not ret: break
    if args["model"] != "1":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    color = get_frame(frame, args["model"])
    # Save the video
    frames.append(color)
    i +=1
cap.release()

out = cv2.VideoWriter(args["output"], fourcc, FPS, (FrameSize), 1) # ATTENTION, DERNIER PARAM : ISCOLOR
for frame in frames:
    out.write(frame)
out.release()
cv2.destroyAllWindows()
