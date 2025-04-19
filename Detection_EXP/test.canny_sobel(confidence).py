import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import math
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
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (w1*h1) + (w2*h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def merge_overlapping_boxes_with_confidence(boxes_with_conf, iou_threshold=0.5):
    merged = True
    while merged:
        merged = False
        new_boxes = []
        while boxes_with_conf:
            boxA, confA = boxes_with_conf.pop()
            has_merged = False
            for i, (boxB, confB) in enumerate(new_boxes):
                if iou(boxA, boxB) > iou_threshold:
                    merged_box = union_box(boxA, boxB)
                    merged_conf = max(confA, confB)
                    new_boxes[i] = (merged_box, merged_conf)
                    has_merged = True
                    merged = True
                    break
            if not has_merged:
                new_boxes.append((boxA, confA))
        boxes_with_conf = new_boxes
    return boxes_with_conf

def difference_based_object_detection(depth_video_path,
                                      rgb_video_path,
                                      frame_gap=1,
                                      overlap_threshold=0.5,
                                      iou_threshold_for_union=0.5,
                                      yolo_model=None,
                                      output_dir='detections',
                                      show_window=True):
    depth_cap = cv2.VideoCapture(depth_video_path)
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    if not depth_cap.isOpened() or not rgb_cap.isOpened():
        print("無法讀取 Depth 或 RGB 視頻，請檢查路徑。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 影片輸出設定
    output_video_path = os.path.join(output_dir, "merged_detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = depth_cap.get(cv2.CAP_PROP_FPS)
    width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frames_buffer = deque(maxlen=frame_gap)
    frame_index = 0

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

        processed_frame = image_process(depth_frame)
        prev_frame = frames_buffer[-frame_gap]
        frame_diff = cv2.absdiff(prev_frame, processed_frame)

        if len(frame_diff.shape) != 2:
            frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = np.clip(frame_diff, 0, 255).astype(np.uint8)
        _, sobel_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        sobel_contours, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sobel_boxes = [cv2.boundingRect(contour) for contour in sobel_contours if cv2.contourArea(contour) >= 10]

        gray_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray_frame, 10, 15)
        canny_boxes = generate_candidate_boxes(canny_edges)

        final_boxes = []
        final_boxes_with_conf = []
        for canny_box in canny_boxes:
            max_overlap = 0
            for sobel_box in sobel_boxes:
                overlap = calculate_overlap_ratio(canny_box, sobel_box)
                max_overlap = max(max_overlap, overlap)
                if overlap >= overlap_threshold:
                    # 不只使用單純的 overlap，結合其他因素
                    x, y, w, h = canny_box
                    area = w * h
                    aspect_ratio = max(w/h, h/w) if h != 0 else 1
                    
                    # 計算形狀因子 (越接近1越好)
                    shape_factor = 1.0 / (aspect_ratio if aspect_ratio >= 1 else 1.0)
                    
                    # 計算面積因子 (太小或太大的框都降低可信度)
                    area_factor = min(1.0, area / 1000) * min(1.0, 10000 / (area + 1))
                    
                    # 結合多種因素計算信心分數
                    conf = overlap * 0.6 + shape_factor * 0.2 + area_factor * 0.2
                    
                    # 確保信心分數在 0~1 範圍內
                    conf = max(0.0, min(1.0, conf))
                    
                    final_boxes_with_conf.append((canny_box, conf))
                    break
            
            # 如果沒有找到匹配的sobel_box，但有一定程度的重疊，仍然加入結果
            if max_overlap > 0.2 and max_overlap < overlap_threshold:
                # 降低信心分數
                conf = max_overlap * 0.4  # 降低信心因子
                final_boxes_with_conf.append((canny_box, conf))

        # 提取 YOLO 檢測結果並加入信心分數
        yolo_boxes_with_conf = []
        if yolo_model is not None:
            yolo_results = yolo_model.predict(rgb_frame, verbose=False, conf=0.50)
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    
                    # 原始YOLO信心分數
                    yolo_conf = float(box.conf[0])
                    
                    # 調整 YOLO 的信心分數，避免過度自信
                    # 當信心太高時做輕微懲罰
                    if yolo_conf > 0.9:
                        adjusted_conf = 0.9 + (yolo_conf - 0.9) * 0.5
                    else:
                        adjusted_conf = yolo_conf
                        
                    # 考慮框的大小因素
                    area = w * h
                    if area < 500:  # 太小的框可能不可靠
                        adjusted_conf *= 0.9
                    elif area > 100000:  # 太大的框也可能有問題
                        adjusted_conf *= 0.95
                        
                    # 收集 YOLO 框及其調整後的信心分數
                    yolo_boxes_with_conf.append(((int(x1), int(y1), int(w), int(h)), adjusted_conf))

        # 合並所有框及其信心分數
        all_boxes_with_conf = final_boxes_with_conf + yolo_boxes_with_conf

        # 在合併重疊框前，對所有信心分數進行歸一化處理
        if all_boxes_with_conf:
            # 提取所有信心分數
            all_confidences = [conf for _, conf in all_boxes_with_conf]
            
            # 如果有足夠多的框，進行標準化處理
            if len(all_confidences) > 5:
                mean_conf = sum(all_confidences) / len(all_confidences)
                # 計算標準差
                std_conf = (sum((c - mean_conf) ** 2 for c in all_confidences) / len(all_confidences)) ** 0.5
                
                if std_conf > 0.01:  # 避免除以接近零的數
                    # Z-分數歸一化
                    normalized_boxes_with_conf = []
                    for box, conf in all_boxes_with_conf:
                        z_score = (conf - mean_conf) / std_conf
                        # sigmoid轉換確保在0~1範圍內，並使分布更均勻
                        norm_conf = 1.0 / (1.0 + math.exp(-z_score))
                        normalized_boxes_with_conf.append((box, norm_conf))
                    all_boxes_with_conf = normalized_boxes_with_conf

        # 合並重疊框
        merged_boxes_with_conf = merge_overlapping_boxes_with_confidence(all_boxes_with_conf, iou_threshold=iou_threshold_for_union)

        # 展示和保存結果
        final_detection_frame = rgb_frame.copy()
        for (x, y, w, h), conf in merged_boxes_with_conf:
            # 依信心分數設置不同顏色
            color = (
                int(255 * (1 - conf)),  # B: 低信心為白色，高信心為紫色
                int(0),                 # G: 固定為0
                int(255)                # R: 固定為紫
            )
            cv2.rectangle(final_detection_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(final_detection_frame, f"{conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 寫入影片
        out_video.write(final_detection_frame)

        # 存檔
        txt_filename = os.path.join(output_dir, f"detections_{frame_index:04d}.txt")
        with open(txt_filename, 'w') as f:
            for (x, y, w, h), conf in merged_boxes_with_conf:
                # 保存框座標及其信心分數
                f.write(f"{x}, {y}, {w}, {h}, {conf:.4f}\n")


        # 視窗顯示（選擇性）
        if show_window:
            combined_frame = depth_frame.copy()
            for x, y, w, h in sobel_boxes:
                cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for x, y, w, h in canny_boxes:
                cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for x, y, w, h in final_boxes:
                cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cv2.imshow("Canny Edges", canny_edges)
            cv2.imshow("Sobel Threshold", sobel_thresh)
            cv2.imshow("Combined (Diff + YOLO)", combined_frame)
            cv2.imshow("Final Merged Boxes", final_detection_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        frames_buffer.append(processed_frame)
        frame_index += 1

    depth_cap.release()
    rgb_cap.release()
    out_video.release()
    if show_window:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    depth_video_path = r"E:\論文\期刊\code\Detection_EXP\video2\test5_240fps_input.mp4"
    rgb_video_path   = r"E:\論文\期刊\code\Detection_EXP\video2\test5_240fps_input.mp4"
    
    model = YOLO(r"E:\論文\期刊\code\YOLO\satellite3_train.pt")

    difference_based_object_detection(
        depth_video_path=depth_video_path,
        rgb_video_path=rgb_video_path,
        frame_gap=1,
        overlap_threshold=0.3,           # 差分+Canny 的框互相比對時用的閾值
        iou_threshold_for_union=0.5,     # 與 YOLO 框合併時的 IOU 閾值
        yolo_model=None,
        output_dir=r"E:\論文\期刊\code\Detection_EXP\video2\detections_T_conf"          # 輸出檔案所在資料夾
    )
