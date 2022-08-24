# Report

**Code Link:** https://github.com/slaughterer24/dino-game-using-hand-gestures/blob/master/play_game.py


First, we import the necessary modules:
```
import pyautogui
import numpy as np
import cv2
```
What our program basically does is that it sees the video feed, looks for a hand, and presses a key when the hand moves accordingly. `pyautogui` is required for pressing the key after detection of that particular movement. `numpy` is used twice in this program only to initialize 2 values (as we will see below). `cv2` is the main module required for all the Computer Vision operations.

```
last = "centre"
downPressed = False
cam = cv2.VideoCapture(0)
```
We declared a string `last` and a boolean `downPressed`, we will see their uses later on. In the line `cam = cv2.VideoCapture(0)`, we have created a `VideoCapture` object, which helps us iterating through each frame of the video feed captured by the camera. The `0` refers to the primary camera (web cam for most devices).

```
while True:
  _, img = cam.read()
  height, width, depth = img.shape
 ```
We then start iterating through each frame.
`cam.read()` returns a tuple containing 2 values, the first is the boolean which denotes whether the frame was read correctly or not, and the other value is the frame image, represented in the format of a 3D matrix (`numpy.ndarray`). Note that any RGB image can be represented as a 2D collection of pixels, where is pixel has 3 values, each one of red, green and blue (also note that it is stored in the order BGR instead of RGB in the `ndarray`). If the image is grayscale, it can be represented as 2D collection of pixels where each pixel has a value between `0` and `255`, `0` being "pure black" and `255` being "pure white". We assume that the frame is read correctly so we store the first boolean in `_`, which is not used again. We store the image of each frame in variable `img`. Further, we store the height, width and number of channels of the frame (though no. of channels won't be used).

```
blur = cv2.GaussianBlur(img, (3, 3), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 10, 60])
upper = np.array([20, 150, 255])
mask = cv2.inRange(hsv, lower, upper)
```
First, we blur the image using Gaussian Blur so that our algorithms can work better as they work on the important elements rather than the noise. In the `cv2.GaussianBlur()` method, our arguments are the image, the kernel size, and the 3rd argument `0` specifies that the values of sigma x (std. deviation in the x-direction) and sigma y are calculated from the kernel. The kernel is a small 2D array, and can be thought of as a "brush" such that when this "brush" sweeps over the image (iterates over the image), each pixel is changed in a particular way, which in this case results in the blurring of the image. To be somewhat more precise, Gaussian Blur uses the kernel to replace the value of the pixel according to its own and the weighted value of adjoining pixels. (3, 3) denotes the size of the blur (array), higher the values, more is the blur effect.
Using `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` we convert the blurred image from RGB color scheme to HSV color scheme, because it is easier to filter out skin color in HSV. We then create the mask for the skin color portions using `cv2.inRange(hsv, lower, upper)`. Here we used 2 `numpy.ndarray`s of length 3, which correspond to orangish tints in HSV, similar to skin color. All colors between these values are "whitened out" in our mask.

So now we have a mask, which is white in all regions where there is the hand (assuming the hand to be the only skin colored object), and black otherwise. Our mask contains a lot of noise, i.e. unwanted places where skin is detected.

```
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
```
So now we denoise the mask. Along with blur, there are 2 more methods for this job, `cv2.dilate()` and `cv2.erode()`, both using the arguments of image, a kernel, and the number of iterations. `cv2.erode()` increases the black area by using the kernel (if majority of pixels in the kernel are black then all are turned black). Similarly, `cv2.dilate()` increases the white area and works in the reverse of erosion. `cv2.medianBlur()` is another kind of blur which takes the parameters of image and kernel size. In medianBlur, each pixel is replaced by the median of all the pixels in the kernel. Blur reduces sharpness of noise and hence helps in thresholding. Blur, erosion and dilation together are thus used to denoise the mask.
Then the threshold is calculated by the method `cv2.threshold()` which takes the arguments of image, the threshold size, a grayscale color value, and a flag (either 0/`cv2.THRESH_BINARY` or 1/`cv2.THRESH_BINARY_INV`). If the flag is `cv2.THRESH_BINARY` (like here), then all the pixels having value > the threshold value are set to the specified color value (255/white) here. The rest are set to 0/black. The vice-versa is true if flag is `cv2.THRESH_BINARY_INV`. Also, the method itself returns tuple instead of the image- the image lies at index 1.

So now we have our final mask `thresh`, which ideally contains only the hand, but at the very least we can almost safely say that if there are more objects then the largest one out of them is the hand.
```
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
if len(cnts) == 0:
  last = "none"
  continue
```
Now, from our mask `thresh`, we get our list of contours (roughly speaking, the borders surrounding each object) using the above line of code. The flag `cv2.RETR_EXTERNAL` means that if, says, the object is of a donut shape, only the outer boundary is considered as the contour. The flag `cv2.CHAIN_APPROX_SIMPLE` means that redundant points are removed (for e.g. using 4 points for a rectangle rather than a large number of points) thus saving memory. Note that the method itself actually gives a tuple, and the list of contours is actually at one of the indices. This index at which the list of contours lies is different in different OpenCV versions, for the (as of now) latest OpenCV 4.x, the index is 0. We store the list of contours in the variable `cnts`. Now, by any chance if there are no contours (for e.g. no object having skin color) then we skip to the next frame directly (before `continue`, we have put the string `last` to be `"none"` - we will see the use of `last` later).

```
max_area = -1
for c in cnts:
  area = cv2.contourArea(c)
  if area > max_area:
    max_area = area
    cnt = c
```
In the above lines of code, we simply find the contour having the largest area, and we assume this is of the hand. We set `cnt` to be this contour.

```
mom = cv2.moments(cnt)
if mom['m00'] != 0:
  cx = int(mom['m10']/mom['m00'])
  cy = int(mom['m01']/mom['m00'])
```

In the above lines of code, we find the centre of the hand. For this we first get the dictionary of moments of the area captured by the contour using `cv2.moments(cnt)`. `m10 = Σxm_i/Σm_i`, which gives the x-coordinate of the centre of mass. Similarly, `m01 = Σym_i/Σm_i`, which gives the y-coordinate of centre of mass.

```
cv2.drawContours(img, [cnt], -1, (122, 122, 0), 2)
cv2.circle(img, (cx, cy), 7, (0, 0, 255), 2)
cv2.putText(img, "Centre", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.line(img, (0, height//2 - 70), (width, height//2 - 70), (255, 255, 255), 5)
cv2.line(img, (0, height//2 + 70), (width, height//2 + 70), (255, 255, 255), 5)
```
We now draw everything in this frame of our output video.
First we draw the contour pf the hand using the method `cv2.drawContours()` methods. It takes in parameters of the image ndarray, the list of all contours being drawn (in our case just one), the index of the contour to draw (if negative index given like in our case then all contours are drawn), the color of the contour, and its thickness in px.
Then we draw a small red circle indicating the centre point. It is done by the method `cv2.circle()` which takes the parameters of the image ndarray, the centre point, its radius in px, its color and its thickness in px.
Then we add the text "Centre" to the image at the centre point using the method `cv2.putText()` which takes parameters as the image, the string to be written, the point at which to begin writing the string, the font (given by `cv2` flags like `cv2.FONT_HERSHEY_COMPLEX`), the scaling factor, text color, and the thickness in px.
Then we draw 2 bold (5px thick) white horizontal lines, one the the top and one at the bottom. They divide the frame into 3 areas: the top, middle and bottom. They have been drawn using the method `cv2.line()` which takes parameters of the image, the first point, the second point, color of the line and the thickness of the line. Here, we are making the top line 70px above the centre and the bottom line 70px below the centre.

```
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
```
Now this is the part where we press the keys, and we will learn the purpose of the variables `last` and `downPressed`. Now we can't just press the up key if we are in the top region because then if we put the hand on the top region for some time, then it will keep pressing the up key the whole time, which we do not want. Same thing for the bottom region. To take care of this, we made a string `last` which will keep the region the centre was in in the last frame (it it initialized with `"none"`). So now I press up only if `last` does not correspond to the top frame. The key is pressed by `pyautogui.press("up")`.
The case for down key is slightly different. In this game, unlike the up key, if the down key is pressed once, then the dino ducks only for a split second, which is pretty much useless. So I made it happen that the down key will remain pressed while the hand is in the bottom region, and we release it once we get out of the bottom region. For this we have the boolean `downPressed` which corresponds to whether the down key is currently pressed or not (initialized as `False`, obviously). If we are in the bottom region and down key is not pressed, then we start pressing the down key (using `pyautogui.keyDown("down")`) and set `downPressed` to true. Also, if we are in the centre region and `downPressed` is `True`, then we put down the key (using `pyautogui.keyUp("down")`) and set `downPressed` to `False`.

```
cv2.imshow("Frame", img)
```
In the above line we output the particular frame which has the hand contour, centre circle, "Centre" and the 2 lines.

```
k = cv2.waitKey(10) & 0xFF
if k == 27:
  break
```
Now, we wait 10ms each frame for a key press. The code for the key pressed is returned by `cv2.waitKey(10)` (the `10` here is for 10ms). We take its last 8 binary digits by doing a bitwise AND of it with `11111111` (i.e. `0xFF`). If that number amounts to 27, then that means we had pressed the escape key. So we break out of the loop.

```
cam.release()
cv2.destroyAllWindows()
```
Finally, now that we are out of the loop and everything is done, we stop taking video input using `cam.release()` and then close all remaining windows (there won't be any, but just to be safe) using `cv2.destroyAllWindows()`.
