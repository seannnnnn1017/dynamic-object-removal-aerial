import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ---------- 合併框相關工具函式 ----------
def iou(boxA, boxB):
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def union_box(boxA, boxB):
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    left   = min(x1, x2)
    top    = min(y1, y2)
    right  = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    return (left, top, right - left, bottom - top)

def merge_overlapping_boxes(boxes, iou_threshold=0.5):
    merged = True
    while merged:
        merged = False
        new_boxes = []
        while boxes:
            boxA = boxes.pop()
            has_merged = False
            for i, boxB in enumerate(new_boxes):
                if iou(boxA, boxB) > iou_threshold:
                    new_boxes[i] = union_box(boxA, boxB)
                    has_merged = True
                    merged = True
                    break
            if not has_merged:
                new_boxes.append(boxA)
        boxes = new_boxes
    return boxes

# ---------- 差分 + Canny + YOLO 核心函式 ----------
def image_process(img):
    """
    使用 Sobel 算子計算梯度
    """
    dstx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dsty = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    dstx = cv2.convertScaleAbs(dstx)
    dsty = cv2.convertScaleAbs(dsty)
    dst = cv2.addWeighted(dstx, 0.5, dsty, 0.5, 0)
    return dst

def generate_candidate_boxes(edge_img):
    """
    根據邊緣圖生成候選框
    """
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes

def calculate_overlap_ratio(box1, box2):
    """
    計算兩個框的重疊區域佔比（IOU）
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 計算交集
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def difference_based_object_detection(depth_video_path,
                                      rgb_video_path,
                                      frame_gap=1,
                                      overlap_threshold=0.5,
                                      iou_threshold_for_union=0.5,
                                      yolo_model=None):
    """
    差分 + Canny 的最終框 與 YOLO 框 做聯集產生最終預測框
    """
    depth_cap = cv2.VideoCapture(depth_video_path)
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    if not depth_cap.isOpened() or not rgb_cap.isOpened():
        print("無法讀取 Depth 或 RGB 視頻，請檢查路徑。")
        return

    frames_buffer = deque(maxlen=frame_gap)

    # 初始化 frames_buffer
    for _ in range(frame_gap):
        ret_depth, depth_frame = depth_cap.read()
        ret_rgb, rgb_frame = rgb_cap.read()
        if not ret_depth or not ret_rgb:
            break
        processed_frame = image_process(depth_frame)
        frames_buffer.append(processed_frame)

    while True:
        ret_depth, depth_frame = depth_cap.read()
        ret_rgb, rgb_frame = rgb_cap.read()
        if not ret_depth or not ret_rgb:
            break

        # ---------------------------
        #   1) 差分 + Canny 部分
        # ---------------------------
        processed_frame = image_process(depth_frame)
        prev_frame = frames_buffer[-frame_gap]
        frame_diff = cv2.absdiff(prev_frame, processed_frame)

        # 確保 frame_diff 是灰階且為 uint8
        if len(frame_diff.shape) != 2:
            frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = np.clip(frame_diff, 0, 255).astype(np.uint8)

        # 二值化
        _, sobel_thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        # Sobel 框
        sobel_contours, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sobel_boxes = [cv2.boundingRect(contour) for contour in sobel_contours if cv2.contourArea(contour) >= 10]

        # Canny 邊緣檢測
        gray_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray_frame, 10, 15)
        canny_boxes = generate_candidate_boxes(canny_edges)

        # 依 overlap_threshold 篩選最終框
        final_boxes = []
        for canny_box in canny_boxes:
            for sobel_box in sobel_boxes:
                overlap_ratio = calculate_overlap_ratio(canny_box, sobel_box)
                if overlap_ratio >= overlap_threshold:
                    final_boxes.append(canny_box)
                    break

        # ---------------------------
        #   2) YOLO 偵測部分
        # ---------------------------
        yolo_boxes = []
        if yolo_model is not None:
            yolo_results = yolo_model.predict(rgb_frame, verbose=False,conf=0.30)  # 可調整 conf=... 等參數
            for result in yolo_results:
                for box in result.boxes:
                    # 取出 YOLO 預測框 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    yolo_boxes.append((int(x1), int(y1), int(w), int(h)))

        # ---------------------------
        #   3) 合併 (Union) final_boxes + yolo_boxes
        # ---------------------------
        # 把兩者裝在一起
        all_boxes = final_boxes + yolo_boxes

        # 利用 merge_overlapping_boxes() 做 union
        merged_boxes = merge_overlapping_boxes(all_boxes, iou_threshold=iou_threshold_for_union)

        # ---------------------------
        #   4) 繪圖顯示
        # ---------------------------
        # A) 先顯示中間結果：Sobel (藍) & Canny(紅) & final_boxes(綠)
        combined_frame = depth_frame.copy()
        for x, y, w, h in sobel_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 藍色
        for x, y, w, h in canny_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 紅色
        for x, y, w, h in final_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠色
        final_detection_frame = rgb_frame.copy()
        # B) 也可顯示 YOLO (黃色)
        #   這裡我們示範在同一張圖顯示 YOLO 的框
        for x, y, w, h in yolo_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 黃色

        # C) 最終合併框 (粉色)，可以繪製在 RGB 影像或另開視窗
 
        for x, y, w, h in merged_boxes:
            cv2.rectangle(final_detection_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # 粉色

        # 顯示視窗
        cv2.imshow("Canny Edges", canny_edges)
        cv2.imshow("Sobel Difference (Threshold)", sobel_thresh)
        cv2.imshow("Intermediate Combined (Diff + YOLO)", combined_frame)
        cv2.imshow("Final Merged Boxes", final_detection_frame)

        frames_buffer.append(processed_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    depth_cap.release()
    rgb_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    depth_video_path = r"E:\論文\期刊\code\final_video\test5_240fps_vis.mp4"
    rgb_video_path   = r"E:\論文\期刊\code\final_video\test5_240fps_input.mp4"
    
    model = YOLO(r"E:\論文\期刊\code\YOLO\satellite3_train.pt")

    difference_based_object_detection(
        depth_video_path=depth_video_path,
        rgb_video_path=rgb_video_path,
        frame_gap=1,
        overlap_threshold=0.1,         # 差分 + Canny 的框互相比對時用的閾值
        iou_threshold_for_union=0.5,   # 與 YOLO 框合併時的 IOU 閾值
        yolo_model=model
    )
