import os
import cv2
from ultralytics import YOLO

def yolo_based_object_detection(
    rgb_video_path,
    yolo_model=None,
    output_dir='yolo_detections',
    show_window=True
):
    """
    使用 YOLO 模型進行物件偵測。
    """
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    if not rgb_cap.isOpened():
        print("無法讀取 RGB 視頻，請檢查路徑。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 影片輸出設定
    output_video_path = os.path.join(output_dir, "yolo_detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret_rgb, rgb_frame = rgb_cap.read()
        if not ret_rgb:
            break

        yolo_boxes = []
        if yolo_model is not None:
            yolo_results = yolo_model.predict(rgb_frame, verbose=False, conf=0.30)
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    yolo_boxes.append((int(x1), int(y1), int(w), int(h)))

        final_detection_frame = rgb_frame.copy()
        for x, y, w, h in yolo_boxes:
            cv2.rectangle(final_detection_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # 寫入影片
        out_video.write(final_detection_frame)

        # 存檔
        txt_filename = os.path.join(output_dir, f"detections_{frame_index:04d}.txt")
        with open(txt_filename, 'w') as f:
            for x, y, w, h in yolo_boxes:
                f.write(f"{x}, {y}, {w}, {h}\n")

        # 視窗顯示（選擇性）
        if show_window:
            cv2.imshow("YOLO Detections", final_detection_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        frame_index += 1

    rgb_cap.release()
    out_video.release()
    if show_window:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rgb_video_path  = r"E:\論文\期刊\code\指標\video0\aligned_video2_input.mp4"

    model = YOLO(r"E:\論文\期刊\code\YOLO\satellite3_train.pt")

    yolo_based_object_detection(
        rgb_video_path=rgb_video_path,
        yolo_model=model,
        output_dir=r"E:\論文\期刊\code\指標\video0\detections_YOLO_ONLY"  # 輸出檔案所在資料夾
    )