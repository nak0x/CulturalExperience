import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
count = 0
os.makedirs("macbooktheo", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f"macbooktheo/img_{count}.jpg", frame)
        print(f"Saved img_{count}")
        count += 1
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

