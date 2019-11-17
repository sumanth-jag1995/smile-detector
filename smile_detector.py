# Face Recognition using Opencv and Viola Jones algorithm

# Importing the libraries
import cv2

# Loading the cascades
face_cascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Detection
def detect(gray, frame):
    '''
    Inputs:   gray    - The grayscale of the original color image
              frame   - The original color image
    Variables:x, y    - The co-ordinates of the upper left corner of the rectangle
              w, h    - Width and Height of the rectangle of the face respectively
              ex, ey  - The co-ordinates of the upper left corner of the rectangle of the eyes
              ew, eh  - Width and Height of the rectangle of the eyes respectively
              sx, sy  - The co-ordinates of the upper left corner of the rectangle of the smile
              sw, sh  - Width and Height of the rectangle of the smile respectively
    Output:   frame   - Image with the detector rectangles
    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:           # For each detected face
        # Paint rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Region of interest for the eyes for both images
        roi_gray = gray[y: y+ h, x:x + w]
        roi_color = frame[y: y+ h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:    # For each detected eyes
            # Paint rectangle around the detected eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:    # For each detected smile
            # Paint rectangle around the detected smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    # Return the color image with the detector rectangles
    return frame

# Face recognition with webcam
video_capture = cv2.VideoCapture(0)                # Turn on webcam
while True:
    _, frame = video_capture.read()                # Latest frame from the webcam, '_' ignores the first return variable from the function
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale of the latest frame
    canvas = detect(gray, frame)                   # image with the detector rectangles
    cv2.imshow('Video', canvas)                    # Display the video with detector rectangles
    if cv2.waitKey(1) & 0xFF == ord('q'):          # Keyboard interrupt
        break

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
        
