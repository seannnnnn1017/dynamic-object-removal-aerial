import cv2
import numpy as np
from collections import deque
import time

def image_process(img):
    # Sobel 邊緣檢測
    dstx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dsty = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    dstx = cv2.convertScaleAbs(dstx)
    dsty = cv2.convertScaleAbs(dsty)
    
    # 加權後的邊緣強化
    combined = cv2.addWeighted(dstx, 0.5, dsty, 0.5, 0)
    return dstx, dsty, combined

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟視頻檔案")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 原始影像
        original_frame = frame.copy()

        # 進行 Sobel 邊緣檢測與加權後的邊緣強化
        sobel_x, sobel_y, combined_sobel = image_process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # 顯示原始影像與處理後的結果
        cv2.imshow("Original Frame", original_frame)
        cv2.imshow("Sobel X", sobel_x)
        cv2.imshow("Sobel Y", sobel_y)
        cv2.imshow("Combined Sobel (Weighted)", combined_sobel)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"E:\論文\期刊\code\指標\video2\test5_480fps_input.mp4"
    process_video(video_path)
