#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/Users/rohinivedam/Documents/Horizon_2.0/EngageVision')
from engagevision.head_pose_estimation import HeadPoseEstimator
from engagevision.gaze_tracking import GazeTracking
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import build_sam, SamPredictor
from torchvision import models, transforms
from typing import List
import os


# In[26]:


import os

checkpoint_path = "./sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt"

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. "
                            "Please download it from the official SAM2 repo or Hugging Face.")

else :
    print(f"Checkpoint found at {checkpoint_path}. Proceeding with loading the model.") 


# In[ ]:


import sys
import torch

import logging
logging.getLogger("hydra").setLevel(logging.WARNING)

# Add the repo root to Python path
sys.path.append("/Users/rohinivedam/Documents/Horizon_2.0/sam2_repo")

# Import SAM2 model builder and video predictor
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor  # use the correct filename

sam = build_sam2(
    config_file=config_file,
    ckpt_path=checkpoint_path,
    device="cpu",            # Force CPU
    mode="eval"
)

def load_hiera_sam_model(config_file, checkpoint_path):
    """
    Load a Hierarchical SAM model for video processing.

    Args:
        config_file (str): Path to the configuration file for the SAM model.
        checkpoint_path (str): Path to the checkpoint file for the SAM model.

    Returns:
        SAM2VideoPredictor: An instance of the SAM2VideoPredictor initialized with the
        Hierarchical SAM model.
    """
    sam = build_sam2(config_file=config_file, ckpt_path=checkpoint_path)
    device = "cpu" 
    sam.to(device)
    predictor = SAM2VideoPredictor(sam)  # or SamPredictor(sam) if single-image
    return predictor

# Example usage
config_file = "configs/sam2.1/sam2.1_hiera_b+.yaml"
checkpoint_path = "./sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt"

predictor = load_hiera_sam_model(config_file, checkpoint_path)


# 

# 

# In[2]:


import sam2
dir(sam2)


# In[33]:


def segment_frame(predictor, frame):
    predictor.set_image(frame)
    H, W = frame.shape[:2]
    center = np.array([[W // 2, H // 2]])
    label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=center,
        point_labels=label,
        multimask_output=False
    )
    return masks[0]


# In[34]:


def extract_keyframes(video_path, predictor, threshold=0.05, stride=5):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    prev_mask = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        mask = segment_frame(predictor, frame)

        if prev_mask is None or np.mean(np.abs(mask - prev_mask)) > threshold:
            keyframes.append(frame)
            prev_mask = mask

        frame_idx += 1

    cap.release()
    return keyframes


# In[35]:


def preprocess_keyframes(keyframes, size=224):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean/std for ImageNet
                             [0.229, 0.224, 0.225])
    ])
    return [transform(kf) for kf in keyframes]


# In[36]:


def normalize_sequence_length(tensor_list, target_length=32):
    seq_len = len(tensor_list)
    C, H, W = tensor_list[0].shape

    if seq_len >= target_length:
        return torch.stack(tensor_list[:target_length])  # [target_length, C, H, W]
    else:
        pad_count = target_length - seq_len
        padding = [tensor_list[-1]] * pad_count
        return torch.stack(tensor_list + padding)



# In[37]:


head_pose_estimator = HeadPoseEstimator()
gaze_tracker = GazeTracking()


# In[ ]:


def extract_engagement_features(frame) -> np.ndarray:
    """
    Extract [gaze_x, gaze_y, yaw, pitch, roll] using EngageVision
    """
    gaze_tracker.refresh(frame)
    head_pose_estimator.refresh(frame)

    # Get gaze direction (x, y)
    gaze_x = gaze_tracker.horizontal_ratio() or 0.0
    gaze_y = gaze_tracker.vertical_ratio() or 0.0

    # Get head pose (yaw, pitch, roll)
    pose = head_pose_estimator.get_angles()
    if pose is None:
        yaw, pitch, roll = 0.0, 0.0, 0.0
    else:
        yaw, pitch, roll = pose

    return np.array([gaze_x, gaze_y, yaw, pitch, roll], dtype=np.float32)


# In[38]:


def build_feature_sequence(video_frames: List[np.ndarray], sequence_length: int = 32) -> torch.Tensor:
    features = [extract_engagement_features(f) for f in video_frames]

    # Pad or truncate to fixed length
    if len(features) >= sequence_length:
        features = features[:sequence_length]
    else:
        last = features[-1] if features else np.zeros(5, dtype=np.float32)
        features += [last] * (sequence_length - len(features))

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, T, D]


# In[ ]:


class EngagementRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)  # Output engagement score (0-1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # [B, T, H]
        out = self.out(rnn_out[:, -1, :])  # last time step
        return torch.sigmoid(out)  # regression score


# In[ ]:


# ----------------------------
# 5. Simulated Pipeline Execution
# ----------------------------
if __name__ == "__main__":
    # Simulate reading 40 frames from webcam or video
    cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for video file
    frames = []
    while len(frames) < 40:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Extract time-series features
    sequence = build_feature_sequence(frames, sequence_length=32)

    # Load model
    model = EngagementRNN()

    # Predict engagement score
    score = model(sequence)
    print(f"Predicted Engagement Score: {score.item():.3f}")

