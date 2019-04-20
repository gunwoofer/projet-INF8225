import numpy as np
import cv2

cap = cv2.VideoCapture('julia.avi')

ret, frame = cap.read()
print('ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])


FPS= 30.0
FrameSize=(frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'XVID')

frames = []
while(cap.isOpened()):
    ret, frame = cap.read()

    # check for successfulness of cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Save the video
    frames.append(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()

out = cv2.VideoWriter('Video_output.avi', fourcc, FPS, FrameSize, 0)
for frame in frames:
	out.write(frame)
out.release()
cv2.destroyAllWindows()