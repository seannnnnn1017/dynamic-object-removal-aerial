import cv2
import numpy as np
from collections import deque
from tqdm import tqdm

def calculate_metrics(all_detections, iou_threshold=0.5):
    """
    計算多個偵測指標，包括 Precision, Recall, F1, Average IoU, 和 mAP@50
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    iou_list = []
    ap_list = []  # 儲存每幀的 AP 用於計算 mAP

    for pred_boxes, gt_boxes in all_detections:
        matches = []
        used_gt_idx = set()  # 用於追踪已匹配的 Ground Truth
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx not in used_gt_idx:  # 確保此 Ground Truth 未被匹配
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matches.append((pred_box, gt_boxes[best_gt_idx], best_iou))
                used_gt_idx.add(best_gt_idx)
                iou_list.append(best_iou)
        
        # 計算 TP, FP, FN
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 計算 Precision 和 Recall 曲線
        precisions, recalls = [], []
        for t in range(0, 101, 10):  # Recall 以 0.1 間隔取點
            recall_threshold = t / 100
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        # 計算每幀的 AP
        ap = 0
        for r in np.linspace(0, 1, 11):  # 11點插值法
            prec_at_r = [p for p, rec in zip(precisions, recalls) if rec >= r]
            if prec_at_r:
                ap += max(prec_at_r)
        ap /= 11
        ap_list.append(ap)

    # 計算總體指標
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(iou_list) if iou_list else 0
    map50 = np.mean(ap_list) if ap_list else 0

    return precision, recall, f1_score, avg_iou, map50
def manual_label_with_diff(frame, diff_frame, pred_boxes,sobel_boxes,canny_boxes):
    """手動標註物件，返回標註的邊界框，同時顯示差分結果的檢測框"""
    boxes = []
    current_box = []
    clone_left = frame.copy()
    clone_right = frame.copy()

    if False:
        for box in sobel_boxes:
            x, y, w, h = box
            cv2.rectangle(clone_right, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 在右側顯示的原影片上繪製預測邊界框
        
        for box in canny_boxes:
            x, y, w, h = box
            cv2.rectangle(clone_right, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 在右側顯示的原影片上繪製預測邊界框
    for box in pred_boxes:
        x, y, w, h = box
        cv2.rectangle(clone_right, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 在右側顯示的原影片上繪製預測邊界框

    combined_display = np.hstack((clone_left, clone_right))  # 並排顯示左側原影片與右側含預測框的原影片

    def draw_rectangle(event, x, y, flags, param):
        nonlocal current_box, clone_left, combined_display
        if event == cv2.EVENT_LBUTTONDOWN:
            current_box = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            current_box.append((x, y))
            x1, y1 = current_box[0]
            x2, y2 = current_box[1]
            boxes.append(convert_to_bbox_format(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            cv2.rectangle(clone_left, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 用藍色框表示標註
            combined_display = np.hstack((clone_left, clone_right))  # 更新顯示
            cv2.imshow("Manual Labeling", combined_display)

    cv2.imshow("Manual Labeling", combined_display)
    cv2.setMouseCallback("Manual Labeling", draw_rectangle)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按 q 結束標註
            break
    cv2.destroyWindow("Manual Labeling")
    return boxes
def convert_to_bbox_format(x1, y1, x2, y2):
    """將 [x1, y1, x2, y2] 格式轉換為 [x, y, w, h] 格式"""
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

def calculate_iou(box1, box2):
    """計算兩個邊界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou

def image_process(img):
    """圖像處理函數，計算 Sobel 梯度"""
    dstx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dsty = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    dstx = cv2.convertScaleAbs(dstx)
    dsty = cv2.convertScaleAbs(dsty)
    dst = cv2.addWeighted(dstx, 0.5, dsty, 0.5, 0)
    return dst

def generate_candidate_boxes(edge_img):
    """根據邊緣圖生成候選框"""
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes

def calculate_overlap_ratio(box1, box2):
    """計算兩個框的重疊區域佔比（IOU）"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 計算交集
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # 沒有交集
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    # 計算交集面積
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 計算聯合區域面積
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # 返回重疊比例（IOU）
    return intersection_area / union_area

def dual_video_object_detection_with_diff(video_path, frame_gap, max_frames=1, overlap_threshold=0.2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: 無法開啟視頻檔案")
        return

    frames_buffer = deque(maxlen=frame_gap)
    all_detections = []  # 儲存所有幀的檢測結果
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("開始處理視頻...")
    frame_count = 0
    with tqdm(total=min(max_frames, total_frames)) as pbar:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = image_process(frame)
            frames_buffer.append(processed_frame)

            if len(frames_buffer) < frame_gap:
                continue

            prev_frame = frames_buffer[-frame_gap]
            frame_diff = cv2.absdiff(prev_frame, processed_frame)

            # 差分處理
            _, sobel_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            if len(sobel_thresh.shape) != 2:
                sobel_thresh = cv2.cvtColor(sobel_thresh, cv2.COLOR_BGR2GRAY)
            sobel_boxes = generate_candidate_boxes(sobel_thresh)

            # Canny 邊緣檢測
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_edges = cv2.Canny(gray_frame, 10, 30)
            canny_boxes = generate_candidate_boxes(canny_edges)

            # 比較重疊框，篩選最終框
            final_boxes = []
            for canny_box in canny_boxes:
                for sobel_box in sobel_boxes:
                    overlap_ratio = calculate_overlap_ratio(canny_box, sobel_box)
                    if overlap_ratio >= overlap_threshold:
                        final_boxes.append(canny_box)
                        break

            # 手動標註
            print(f"請手動標註第 {frame_count + 1} 幀的物件，按 Q 結束標註...")
            gt_boxes = manual_label_with_diff(frame, sobel_thresh, final_boxes,sobel_boxes,canny_boxes)

            # 儲存檢測結果
            all_detections.append((final_boxes, gt_boxes))
            frame_count += 1
            pbar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    precision, recall, f1_score, avg_iou, map50 = calculate_metrics(all_detections)

    print("\n評估結果:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"mAP@50: {map50:.4f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"images\test.mp4"
    dual_video_object_detection_with_diff(video_path, frame_gap=10, max_frames=5)
