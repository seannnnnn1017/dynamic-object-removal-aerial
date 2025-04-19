import cv2
import numpy as np
from collections import deque
import time
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

def difference_based_object_detection(video_path,RGB_video_path, frame_gap, overlap_threshold=0.5):
    Depth_cap = cv2.VideoCapture(video_path)
    RGB_cap = cv2.VideoCapture(RGB_video_path)
    if not Depth_cap.isOpened():
        print("無法讀取視頻")
        return
    frames_buffer = deque(maxlen=frame_gap)

    # 初始化緩衝區
    for _ in range(frame_gap):
        ret_depth, Depth_frame = Depth_cap.read()
        ret_rgb, RGB_frame = RGB_cap.read()

        if not ret_depth or not ret_rgb:
            break
        processed_frame = image_process(Depth_frame)
        frames_buffer.append(processed_frame)

    while True:
        ret, Depth_frame = Depth_cap.read()
        ret, RGB_frame = RGB_cap.read()
        if not ret:
            break

        # Sobel 差分處理
        processed_frame = image_process(Depth_frame)
        prev_frame = frames_buffer[-frame_gap]
        frame_diff = cv2.absdiff(prev_frame, processed_frame)

        # 確保 frame_diff 是灰階且為 uint8
        if len(frame_diff.shape) != 2:
            frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = np.clip(frame_diff, 0, 255).astype(np.uint8)

        # 二值化處理
        _, sobel_thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        # Sobel 框生成
        sobel_contours, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sobel_boxes = [cv2.boundingRect(contour) for contour in sobel_contours if cv2.contourArea(contour) >= 100]

        # Canny 邊緣檢測
        gray_frame = cv2.cvtColor(Depth_frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray_frame, 10, 30)
        canny_boxes = generate_candidate_boxes(canny_edges)

        # 比較重疊框，根據重疊閥值篩選
        final_boxes = []
        for canny_box in canny_boxes:
            for sobel_box in sobel_boxes:
                overlap_ratio = calculate_overlap_ratio(canny_box, sobel_box)
                if overlap_ratio >= overlap_threshold:  # 如果重疊比例超過閥值
                    final_boxes.append(canny_box)
                    break

        # 繪製候選框
        combined_frame = Depth_frame.copy()
        final_frame = Depth_frame.copy()

        # 繪製 Canny 框（紅色）
        for x, y, w, h in sobel_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 紅色框

        # 繪製 Canny 框（紅色）
        for x, y, w, h in canny_boxes:
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 紅色框

        # 繪製最終框（綠色）
        for x, y, w, h in final_boxes:
            cv2.rectangle(RGB_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠色框

        # 顯示 Canny 邊緣圖和 Sobel 差分圖
        cv2.imshow("Canny Edges", canny_edges)
        cv2.imshow("Sobel Difference", sobel_thresh)

        # 顯示綜合檢測結果
        cv2.imshow("Combined Detection (Sobel & Canny)", combined_frame)
        cv2.imshow("Final Detected Objects (Green Only)", RGB_frame)

        frames_buffer.append(processed_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    Depth_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Depth_video_path = r"E:\論文\期刊\code\final_video\aligned_tests_cropped_output_depth.mp4"
    RGB_video_path = r"E:\論文\期刊\code\final_video\aligned_tests_cropped_output_input.mp4"
    difference_based_object_detection(Depth_video_path,RGB_video_path, frame_gap=1, overlap_threshold=0.3)