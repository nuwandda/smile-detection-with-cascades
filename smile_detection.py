import cv2

# Load the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# A function that detects face and the smile
def detect(frame, gray):
    # These parameters are decided after some tests and there are the best fits
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Get only the face from the detection
        # To improve the performance, we are bounding the search area.
        # We will only search inside the face
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # Lower number of neighbours means it will detect everthing that is similiar to a a smile
        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (ex, ey, ew, eh) in smile:
            # Add text if there is a smile
            cv2.putText(roi_color, 'Smiling', (10, int(ew / 2)), font, 2, (255,255,255), 2, cv2.LINE_AA)

    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(frame, gray)

    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
