import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from moviepy import ImageSequenceClip, TextClip, CompositeVideoClip, concatenate_videoclips

# ===== 設定 =====
RAW_DIR = "images/raw"        # 入力写真
CROP_DIR = "images/cropped"   # 顔切り出し後
VIDEO_DIR = "videos/individual"
FINAL_VIDEO = "videos/final.mp4"
CSV_FILE = "students.csv"     # 児童リストCSV

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


# ===== 顔切り出し =====
def crop_face(image_path, save_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"[手動クロップ開始] {image_path}")
        roi = cv2.selectROI("Select Face", img, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        if w == 0 or h == 0:
            print("❌ 選択キャンセル、スキップ")
            cv2.destroyAllWindows()
            return False
        face_img = img[y:y+h, x:x+w]
    else:
        h, w, _ = img.shape
        landmarks = results.multi_face_landmarks[0].landmark
        x_min = min([lm.x for lm in landmarks]) * w
        x_max = max([lm.x for lm in landmarks]) * w
        y_min = min([lm.y for lm in landmarks]) * h
        y_max = max([lm.y for lm in landmarks]) * h
        margin = 0.2
        x1 = max(int(x_min - margin*(x_max-x_min)), 0)
        y1 = max(int(y_min - margin*(y_max-y_min)), 0)
        x2 = min(int(x_max + margin*(x_max-x_min)), w)
        y2 = min(int(y_max + margin*(y_max-y_min)), h)
        face_img = img[y1:y2, x1:x2]

    face_img = cv2.resize(face_img, (512, 512))
    cv2.imwrite(save_path, face_img)
    cv2.destroyAllWindows()
    return True


# ===== モーフィング =====
def morph_images(img1, img2, steps=30):
    frames = []
    for alpha in np.linspace(0, 1, steps):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


# ===== 個別動画作成 =====
def make_video(name, class_name, child_msg, parent_msg):
    img1_path = os.path.join(CROP_DIR, f"{name}_1.jpg")
    img2_path = os.path.join(CROP_DIR, f"{name}_2.jpg")
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"❌ {name} の画像不足")
        return None

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    frames = morph_images(img1, img2, steps=40)

    clip = ImageSequenceClip(frames, fps=15)

    # メッセージを重ねる
    txt1 = TextClip(child_msg, fontsize=32, color="white", size=(512, 100), method="caption").set_duration(2)
    txt2 = TextClip(parent_msg, fontsize=32, color="yellow", size=(512, 100), method="caption").set_duration(2)
    txt = concatenate_videoclips([txt1, txt2]).set_position(("center", "bottom"))

    video = CompositeVideoClip([clip, txt])
    out_path = os.path.join(VIDEO_DIR, f"{class_name}_{name}.mp4")
    video.write_videofile(out_path, codec="libx264", fps=15)
    return out_path


# ===== 全員分処理 =====
def main():
    videos = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            class_name = row["class"]
            child_msg = row["child_msg"]
            parent_msg = row["parent_msg"]

            # 顔切り出し（1年, 6年）
            raw1 = os.path.join(RAW_DIR, f"{name}_1.jpg")
            raw2 = os.path.join(RAW_DIR, f"{name}_2.jpg")
            crop1 = os.path.join(CROP_DIR, f"{name}_1.jpg")
            crop2 = os.path.join(CROP_DIR, f"{name}_2.jpg")
            if not os.path.exists(crop1):
                crop_face(raw1, crop1)
            if not os.path.exists(crop2):
                crop_face(raw2, crop2)

            # 動画作成
            video_path = make_video(name, class_name, child_msg, parent_msg)
            if video_path:
                videos.append(video_path)

    # 全員分を結合
    clips = [ImageSequenceClip([], fps=15).set_duration(0)]  # dummy
    for v in videos:
        clips.append(VideoFileClip(v))
    final = concatenate_videoclips(clips[1:])
    final.write_videofile(FINAL_VIDEO, codec="libx264", fps=15)


if __name__ == "__main__":
    main()
