import cv2

# 讀取影片
video_path = r"E:\論文\期刊\code\240fps\test2_240fps.mp4"
cap = cv2.VideoCapture(video_path)

# MOG2 方法來偵測移動物件
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_interval = max(1, total_frames // 5)  # 取 5 張影格來觀察效果

frame_idx = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if frame_idx % sample_interval == 0:
        fgmask = fgbg.apply(frame)  # 計算前景遮罩
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # 二值化處理
        fgmask = cv2.medianBlur(fgmask, 5)  # 去除雜訊

        # 顯示原始影格
        cv2.imshow("Original Frame", frame)

        # 顯示 MOG2 前景遮罩
        cv2.imshow("MOG2 Foreground Mask", fgmask)

        frame_count += 1

        # 按 'q' 退出，按其他鍵繼續
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
  