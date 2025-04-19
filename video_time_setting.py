import cv2

def align_videos_same_length(
    video_path1, 
    video_path2, 
    output_path1="aligned_video1.mp4", 
    output_path2="aligned_video2.mp4"
):
    """ 
    讀取兩部影片，裁剪到相同的影格數，並同時以相同的 FPS 寫出，
    讓輸出後的兩支影片「播放時間」與「總影格數」都完全相同。
    """

    # 1. 開啟影片 1
    cap1 = cv2.VideoCapture(video_path1)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2. 開啟影片 2
    cap2 = cv2.VideoCapture(video_path2)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3. 找出可以保留的最小影格數，並決定最終的輸出 FPS（簡單用 min）
    min_frames = min(frame_count1, frame_count2)
    final_fps = min(fps1, fps2)

    print(f"Video1: {video_path1}")
    print(f"  Original: FPS={fps1}, Frames={frame_count1}, Size=({w1}x{h1})")
    print(f"Video2: {video_path2}")
    print(f"  Original: FPS={fps2}, Frames={frame_count2}, Size=({w2}x{h2})")
    print(f"\n=> 將裁剪到相同的影格數: {min_frames}")
    print(f"=> 最終輸出 FPS: {final_fps}\n")

    # 4. 建立輸出物件（這裡不做 resize，直接保留各自原尺寸）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(output_path1, fourcc, final_fps, (w1, h1))
    out2 = cv2.VideoWriter(output_path2, fourcc, final_fps, (w2, h2))

    # 5. 逐幀讀取並各自僅保留 min_frames 幀
    #    為了方便，先單純 sequential 讀取到 min_frames 為止
    count1 = 0
    while count1 < min_frames:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        out1.write(frame1)
        count1 += 1

    count2 = 0
    while count2 < min_frames:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        out2.write(frame2)
        count2 += 1

    # 6. 收尾
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    print("對齊完成！")
    print(f"輸出影片 1: {output_path1}")
    print(f"輸出影片 2: {output_path2}")

if __name__ == "__main__":
    # 範例路徑：請自行替換
    video1_path = r"aligned_output\aligned_test2_cropped_output.mp4"
    video2_path = r"aligned_output\aligned_test2_depth.mp4"

    align_videos_same_length(
        video_path1=video1_path,
        video_path2=video2_path,
        output_path1="aligned_video1.mp4",
        output_path2="aligned_video1_depth.mp4"
    )


