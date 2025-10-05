import os
from venv import logger
import cv2
import csv
import numpy as np
# from moviepy.editor import (
from moviepy import (
    ImageSequenceClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
    AudioClip,
)

CROP_DIR = "cropped"
VIDEO_DIR = "videos"
CSV_FILE = "students.csv"

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ===== 画像モーフィング処理 =====
def morph_images(img1, img2, steps=30):
    frames = []
    for i in range(steps):
        alpha = i / float(steps - 1)
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame[:, :, ::-1])  # BGR→RGB
    return frames

# ===== 個別動画作成 =====
def make_video(name, class_name, child_msg, parent_msg, photo1, photo2):
    if not os.path.exists(photo1) or not os.path.exists(photo2):
        print(f"❌ {name} の画像が見つかりません")
        return None

    img1 = cv2.imread(photo1)
    img2 = cv2.imread(photo2)

    # サイズを合わせる
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    frames = morph_images(img1, img2, steps=40)
    clip = ImageSequenceClip(frames, fps=15)

    # テキストを生成
    def make_text_clip(text, color, duration=2):
        return TextClip(
            text=text,
            method="caption",
            size=(512, 100),
            color=color,
            font='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        ).with_duration(duration)

    txt1 = make_text_clip(child_msg, "white")
    txt2 = make_text_clip(parent_msg, "yellow")
    txt = concatenate_videoclips([txt1, txt2]).with_position(("center", "bottom"))

    video = CompositeVideoClip([clip, txt])
    out_path = os.path.join(VIDEO_DIR, f"{class_name}_{name}.mp4")

    duration = video.duration
    audio = AudioClip(lambda t: np.zeros_like(t), duration=duration)
    video = video.set_audio(audio)
    # 🎬 出力処理を追加
    video.write_videofile(out_path, codec="libx264", fps=24, audio=False,)


    return out_path



# ===== メイン処理 =====
def main():
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            class_name = row["class"].strip()
            child_msg = row["child_msg"].strip()
            parent_msg = row["parent_msg"].strip()
            photo1 = row["photo1"].strip()
            photo2 = row["photo2"].strip()

            print(f"[処理中] {name} ({class_name})")
            make_video(name, class_name, child_msg, parent_msg, photo1, photo2)

if __name__ == "__main__":
    main()
