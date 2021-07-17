import cv2
import time
import numpy as np

video = cv2.VideoWriter_fourcc(*'MP4V')

output_file = cv2.VideoWriter('output.mp4', video, 20.0, (640, 480))

capture = cv2.VideoCapture(0)

start_time = time.sleep(2)

bg = 0

for i in range(60):
    ret, bg = capture.read()

bg = np.flip(bg, axis = 1)

while(capture.isOpened()):
    ret, img = capture.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)
    new_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])

    mask_1 = cv2.inRange(new_image, lower_red, upper_red)

    lower_red_new = np.array([170, 120, 70])
    upper_red_new = np.array([180, 255, 255])

    mask_2 = cv2.inRange(new_image, lower_red_new, upper_red_new)

    mask_1 = mask_1 + mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    res_without_red = cv2.bitwise_and(img, img, mask = mask_2)
    bg_without_red = cv2.bitwise_and(bg, bg, mask = mask_1)

    outcome = cv2.addWeighted(res_without_red, 1, bg_without_red, 1, 0)

    output_file.write(outcome)
    cv2.imshow('Magic', outcome)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()