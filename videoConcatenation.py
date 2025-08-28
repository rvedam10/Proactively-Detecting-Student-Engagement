import pandas as pd
import os
import cv2
import itertools
import re

# -------------------- PARAMETERS --------------------
excel_path = "video_labels.xlsx"
video_dir = "videos/"
output_dir = "concatenated_videos/"
os.makedirs(output_dir, exist_ok=True)

engagement_levels = ["barely_engaged", "engaged", "highly_engaged"]
segment_length_sec = 5  # duration per engagement segment in seconds

# -------------------- READ EXCEL --------------------
df = pd.read_excel(excel_path)

# -------------------- GROUP VIDEOS BY SUBJECT --------------------
subject_videos = {}

for idx, row in df.iterrows():
    video_name = row['video_name']
    label = row['engagement_label']
    
    # Extract subject name from filename (everything before first underscore)
    subject_match = re.match(r"([a-zA-Z0-9]+)_.*\.mp4", video_name)
    if subject_match:
        subject = subject_match.group(1)
        if subject not in subject_videos:
            subject_videos[subject] = {lvl: [] for lvl in engagement_levels}
        subject_videos[subject][label].append(video_name)

# -------------------- HELPER FUNCTIONS --------------------
def read_video_segment(video_path, segment_length_sec, fps, width, height):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames_needed = int(segment_length_sec * fps)
    count = 0
    while count < total_frames_needed:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def concatenate_video_segments(video_paths, output_path):
    first_cap = cv2.VideoCapture(os.path.join(video_dir, video_paths[0]))
    width  = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(first_cap.get(cv2.CAP_PROP_FPS))
    first_cap.release()

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for vid in video_paths:
        frames = read_video_segment(os.path.join(video_dir, vid), segment_length_sec, fps, width, height)
        for frame in frames:
            out.write(frame)

    out.release()

# -------------------- CREATE ALL POSSIBLE CONCATENATED VIDEOS --------------------
for subject, vids in subject_videos.items():
    if all(vids[level] for level in engagement_levels):
        # Generate all combinations of one video per engagement level
        all_combinations = itertools.product(*[vids[level] for level in engagement_levels])
        for i, combo in enumerate(all_combinations):
            output_path = os.path.join(output_dir, f"{subject}_concat_{i+1}.mp4")
            concatenate_video_segments(combo, output_path)
            print(f"Saved concatenated video for {subject} [{i+1}]: {combo}")
    else:
        print(f"Skipping {subject}, missing some engagement levels")
