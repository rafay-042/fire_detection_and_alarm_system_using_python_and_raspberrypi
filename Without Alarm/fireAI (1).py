import cv2
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

# Load the Haar cascades for fire detection
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect fire in the frame
    fire = fire_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the fire
    for (x,y,w,h) in fire:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        print("FIRE DETECTED")

    # Display the resulting frame
    cv2.imshow('Fire Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
camera.release()
cv2.destroyAllWindows()