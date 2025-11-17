# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## PROGRAM :

```
Developed by : MADHUVATHANI V
Reg NO : 212223040107
```
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
```
```
model = cv2.imread('Photo.jpg',0)
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.jpeg',0)
```
```
plt.imshow(model,cmap='gray')
plt.show()
```
```
plt.imshow(withglass,cmap='gray')
plt.show()
```
```
plt.imshow(group,cmap='gray')
plt.show()
```
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
```
def detect_face(img):
  face_img = img.copy()
  face_rects = face_cascade.detectMultiScale(face_img)   
  for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
    return face_img
```
```
result = detect_face(withglass)
```
```
plt.imshow(result,cmap='gray')
plt.show()
```
```
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```
```
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
```
```
result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
```
```
eyes = eye_cascade.detectMultiScale(withglass) 
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```
```
cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()
ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')
while True:
    ret, frame = cap.read(0)
    frame = detect_face(frame)
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(300)

cap.release()
plt.close()
```

## OUTPUT:

![image](https://github.com/user-attachments/assets/396c2a48-d168-40a9-832f-0f892c42522c)
![image](https://github.com/user-attachments/assets/ba4b3591-4753-43dc-a93e-9163445fe158)

<img width="1092" height="553" alt="image" src="https://github.com/user-attachments/assets/2f04d2e7-69c6-4c81-baee-b49b927df708" />

![image](https://github.com/user-attachments/assets/816def5e-79e7-45ff-a5ae-13be3b3e0ef3)


## RESULT :
Thus face detection using haar cascades executed sucessfully.
