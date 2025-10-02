import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
from moviepy.editor import ImageSequenceClip, TextClip, CompositeVideoClip, concatenate_videoclips
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        h, w, _ = image.shape
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks)

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect, t2_rect, t_rect = [], [], []
    for i in range(3):
        t1_rect.append((t1[i][0]-r1[0], t1[i][1]-r1[1]))
        t2_rect.append((t2[i][0]-r2[0], t2[i][1]-r2[1]))
        t_rect.append((t[i][0]-r[0], t[i][1]-r[1]))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1,1,1), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    size = (r[2], r[3])
    warp_img1 = cv2.warpAffine(img1_rect, cv2.getAffineTransform(np.float32(t1_rect), np.float32(t_rect)), size)
    warp_img2 = cv2.warpAffine(img2_rect, cv2.getAffineTransform(np.float32(t2_rect), np.float32(t_rect)), size)

    img_rect = (1-alpha)*warp_img1 + alpha*warp_img2
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]*(1-mask) + img_rect*mask

def morph_images(img1, img2, points1, points2, alpha):
    points = ((1-alpha)*points1 + alpha*points2).astype(np.int32)
    img_morph = np.zeros_like(img1)
    tri = Delaunay(points)
    for tri_indices in tri.simplices:
        x,y,z = tri_indices
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha)
    return img_morph

def make_video(photo1_path, photo2_path, msg_child, msg_parent, out_path):
    img1 = cv2.imread(photo1_path)
    img2 = cv2.imread(photo2_path)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)
    if points1 is None or points2 is None:
        print(f"顔検出失敗: {photo1_path}, {photo2_path}")
        return None

    frames = []
    n_frames = 40
    for i in range(n_frames+1):
        alpha = i/n_frames
        morphed = morph_images(img1, img2, points1, points2, alpha)
        frames.append(cv2.cvtColor(morphed, cv2.COLOR_BGR2RGB))

    clip = ImageSequenceClip(frames, fps=15)

    # メッセージをオーバーレイ
    txt1 = TextClip(msg_child, fontsize=40, color='white', size=clip.size, method="caption").set_duration(clip.duration/2).set_position(("center","bottom"))
    txt2 = TextClip(msg_parent, fontsize=40, color='yellow', size=clip.size, method="caption").set_duration(clip.duration/2).set_start(clip.duration/2).set_position(("center","bottom"))

    final = CompositeVideoClip([clip, txt1, txt2])
    final.write_videofile(out_path, codec="libx264", audio=False)
    return final

if __name__ == "__main__":
    df = pd.read_csv("students.csv")
    os.makedirs("videos", exist_ok=True)

    all_clips = []
    for _, row in df.iterrows():
        out_file = f"videos/{row['name']}_{row['class']}.mp4"
        print(f"生成中: {out_file}")
        clip = make_video(row["photo1"], row["photo2"], row["msg_child"], row["msg_parent"], out_file)
        if clip:
            all_clips.append(clip)

    # 全員分を結合
    if all_clips:
        final_video = concatenate_videoclips(all_clips, method="compose")
        final_video.write_videofile("final_video.mp4", codec="libx264", audio=False)
        print("✅ 全員分の上映用動画を final_video.mp4 として出力しました！")
