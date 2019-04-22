import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="./paysage_test.avi",
	help="path to  video")
ap.add_argument("-o", "--output", type=str, default="./output/output.avi",
	help="path to output")

args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["input"])

ret, frame = cap.read()
print('ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])


FPS= 30.0
FrameSize=(frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

frames = []
while(cap.isOpened()):
    ret, frame = cap.read()

    # check for successfulness of cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Save the video
    frames.append(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()

out = cv2.VideoWriter(args["output"], fourcc, FPS, FrameSize, 0)
for frame in frames:
	out.write(frame)
out.release()
cv2.destroyAllWindows()