import numpy as np
import cv2
from main import get_frame
cap = cv2.VideoCapture('julia.avi')

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
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    color = get_frame(gray)
    # Save the video
    frames.append(color)
    i +=1
cap.release()

out = cv2.VideoWriter('Video_couleur_output.avi', fourcc, FPS, (FrameSize), 1) # ATTENTION, DERNIER PARAM : ISCOLOR
for frame in frames:
    out.write(frame)
out.release()
cv2.destroyAllWindows()
