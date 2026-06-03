import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} encontrada!")
    else:
        print(f"Camera {i} NÃO encontrada!")
    cap.release()
