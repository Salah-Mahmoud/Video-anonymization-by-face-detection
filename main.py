from model import detect_faces
import cv2




# Open the default camera (usually the integrated webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, img = cap.read()

    # Detect faces in the frame
    faces = detect_faces(img)

    # Blur faces in the frame
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        blurred_face_roi = cv2.GaussianBlur(face_roi, (23, 23),
                                            30)  # Adjust the kernel size and blur intensity as needed
        img[y:y + h, x:x + w] = blurred_face_roi

    # Display the output in a window
    cv2.imshow('Video', img)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
