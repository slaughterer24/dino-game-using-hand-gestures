import pyautogui
import numpy as np
import cv2

last = "centre"
downPressed = False
cam = cv2.VideoCapture(0)

while True:
  _, img = cam.read()
  height, width, depth = img.shape

  blur = cv2.GaussianBlur(img, (3, 3), 0)
  hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
  
  lower = np.array([0, 10, 60])
  upper = np.array([20, 150, 255])
  mask = cv2.inRange(hsv, lower, upper)

  # Filtering background noise
  kernel_square = np.ones((11, 11), np.uint8)
  kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask = cv2.dilate(mask, kernel_ellipse, iterations=1)
  mask = cv2.erode(mask, kernel_square, iterations=1)
  mask = cv2.dilate(mask, kernel_ellipse, iterations=1)
  mask = cv2.medianBlur(mask, 5)
  kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
  mask = cv2.dilate(mask, kernel_ellipse, iterations=1)
  mask = cv2.medianBlur(mask, 5)

  thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

  # Finding the largest contour (of the hand)
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
  if len(cnts) == 0:
    last = "none"
    continue
  max_area = -1
  for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:
      max_area = area
      cnt = c
  
  # Finding centre of the contour
  mom = cv2.moments(cnt)
  if mom['m00'] != 0:
    cx = int(mom['m10']/mom['m00'])
    cy = int(mom['m01']/mom['m00'])
  
  # Drawing everything in the image
  cv2.drawContours(img, [cnt], -1, (122, 122, 0), 2)
  cv2.circle(img, (cx, cy), 7, (0, 0, 255), 2)
  cv2.putText(img, "Centre", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
  cv2.line(img, (0, height//2 - 70), (width, height//2 - 70), (255, 255, 255), 5)
  cv2.line(img, (0, height//2 + 70), (width, height//2 + 70), (255, 255, 255), 5)

  # pressing the key
  if cy < height//2 - 70:
    if last != "up":
      pyautogui.press("up")
    last = "up"
  elif cy > height//2 + 70:
    if last != "down" and not downPressed:
      pyautogui.keyDown("down")
      downPressed = True
    last = "down"
  else:
    if downPressed:
      pyautogui.keyUp("down")
      downPressed = False
    last = "centre"
  
  cv2.imshow("Frame", img)
  cv2.imshow("Thresh", thresh)
  k = cv2.waitKey(10) & 0xFF
  if k == 27:
    break

cam.release()
cv2.destroyAllWindows()
