import os
import cv2
from ultralytics import YOLO

def yolo_object_detection(rgb_video_path,
                          yolo_model=None,
                          output_dir=r'E:\論文\期刊\code\Detection_EXP\video2\detection_YOLO_conf',
                          show_window=True):
    """
    使用YOLO模型進行RGB視頻的物體檢測。
    """
    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        print("無法讀取視頻，請檢查路徑。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 影片輸出設定
    output_video_path = os.path.join(output_dir, "yolo_detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break

        # 使用YOLO進行物體檢測
        if yolo_model is not None:
            yolo_results = yolo_model.predict(rgb_frame, verbose=False, conf=0.10)
            final_boxes = []
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    # 收集檢測框座標和信心分數
                    final_boxes.append((int(x1), int(y1), int(w), int(h), float(box.conf[0])))

            # 在影像上畫出檢測框
            for (x, y, w, h, conf) in final_boxes:
                color = (0, 255, 0)  # 使用綠色標示框
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(rgb_frame, f"{conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 寫入影片
        out_video.write(rgb_frame)

        # 存檔
        txt_filename = os.path.join(output_dir, f"detections_{frame_index:04d}.txt")
        with open(txt_filename, 'w') as f:
            for (x, y, w, h, conf) in final_boxes:
                f.write(f"{x}, {y}, {w}, {h}, {conf:.4f}\n")

        # 視窗顯示（選擇性）
        if show_window:
            cv2.imshow("YOLO Object Detection", rgb_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        frame_index += 1

    cap.release()
    out_video.release()
    if show_window:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rgb_video_path = r"E:\論文\期刊\code\Detection_EXP\video2\test5_240fps_input.mp4"
    yolo_model = YOLO(r"E:\論文\期刊\code\YOLO\satellite3_train.pt")  # 加載YOLO模型（請確保模型權重路徑正確）
    yolo_object_detection(rgb_video_path, yolo_model)
