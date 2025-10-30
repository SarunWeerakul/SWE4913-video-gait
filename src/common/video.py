import cv2
def fps_of(path): cap=cv2.VideoCapture(path); f=cap.get(cv2.CAP_PROP_FPS) or 0.0; cap.release(); return f
