import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import mediapipe as mp

# -------------------- MediaPipe Setup (for efficiency) --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

# -------------------- Segment frame --------------------
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

# -------------------- Extract keyframes --------------------
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

# -------------------- Preprocess frames --------------------
def get_transform(size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# -------------------- Normalize sequence length --------------------
def normalize_sequence_length(tensor_list, target_length=32):
    seq_len = len(tensor_list)
    
    if seq_len >= target_length:
        return torch.stack(tensor_list[:target_length])
    else:
        pad_count = target_length - seq_len
        padding = [tensor_list[-1]] * pad_count
        return torch.stack(tensor_list + padding)

# -------------------- Headpose & Gaze Feature Extraction (your code) --------------------
def get_headpose_gaze_features(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)

    feature_vector = np.zeros(7, dtype=np.float32)

    if not results.multi_face_landmarks:
        return feature_vector

    for face_landmarks in results.multi_face_landmarks:
        face_2d = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark])
        
        # Landmarks for head pose calculation
        face_2d_one = np.array([face_2d[idx] for idx in [33, 263, 1, 61, 291, 199]], dtype=np.float64)
        face_3d_one = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark], dtype=np.float64)
        
        # Camera matrix and distortion coefficients
        focal_length = 1 * frame.shape[1]
        cam_matrix = np.array([[focal_length, 0, frame.shape[0] / 2],
                               [0, focal_length, frame.shape[1] / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP for head pose
        success, rot_vec, trans_vec = cv2.solvePnP(
            np.array([
                [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0], 
                [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
            ], dtype=np.float64), 
            face_2d_one, cam_matrix, dist_matrix)
        
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x_pose = angles[0] * 360 # Pitch
        y_pose = angles[1] * 360 # Yaw
        z_pose = angles[2] * 360 # Roll
        feature_vector[0:3] = [x_pose, y_pose, z_pose]

        # Gaze Score Calculation
        lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0]) if (face_2d[243,0] - face_2d[130,0]) != 0 else 0
        ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1]) if (face_2d[23,1] - face_2d[27,1]) != 0 else 0
        rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0]) if (face_2d[359,0] - face_2d[463,0]) != 0 else 0
        ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1]) if (face_2d[253,1] - face_2d[257,1]) != 0 else 0
        feature_vector[3:7] = [lx_score, ly_score, rx_score]

        break # Only process the first detected face

    return feature_vector

# -------------------- CNN + headpose feature extraction --------------------
def extract_features_from_segment(frame, cnn_model, transform):
    # CNN features
    input_tensor = transform(frame).unsqueeze(0)
    cnn_feat = cnn_model(input_tensor).squeeze(0)

    # Head pose + gaze features
    hg_feat = get_headpose_gaze_features(frame)
    hg_feat = torch.tensor(hg_feat, dtype=torch.float32)

    # Concatenate CNN + headpose/gaze
    feature_vector = torch.cat([cnn_feat, hg_feat], dim=0)
    return feature_vector

# -------------------- LSTM Model --------------------
class EngagementLSTM(nn.Module):
    def __init__(self, input_dim=519, hidden_dim=256, num_layers=2, num_classes=2):
        super(EngagementLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

# -------------------- Full video -> feature sequence --------------------
def process_video(video_path, predictor, cnn_model, target_length=32):
    transform = get_transform()
    keyframes = extract_keyframes(video_path, predictor)
    features = [extract_features_from_segment(kf, cnn_model, transform)
                for kf in keyframes]
    return normalize_sequence_length(features, target_length)

# -------------------- Example of training loop --------------------
def train_model(videos, labels, predictor, cnn_model, num_epochs=10, batch_size=4, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = EngagementLSTM().to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    lstm_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            sequences = []
            for video_path in batch_videos:
                seq = process_video(video_path, predictor, cnn_model)
                sequences.append(seq)
            sequences = torch.stack(sequences).to(device)

            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = lstm_model(sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(videos):.4f}")
    return lstm_model

# -------------------- Example Usage (requires pre-trained models) --------------------
if __name__ == '__main__':
    # Placeholder for a pre-trained segmentation model like SAM
    # You would need to load this from its respective library (e.g., segment-anything)
    # from segment_anything import SamPredictor, sam_model_registry
    # sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    # predictor = SamPredictor(sam)
    
    # Placeholder for a pre-trained CNN model (e.g., ResNet-18)
    cnn_model = models.resnet18(pretrained=True)
    cnn_model.fc = nn.Identity()  # Remove the final classification layer
    cnn_model.eval()

    # Create dummy data for demonstration
    video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
    labels = [0, 1, 0] # 0 for Engaged, 1 for Disengaged
    
    # Run the training loop
    # trained_model = train_model(video_paths, labels, predictor, cnn_model)
    # print("Model training complete!")
    
# -------------------- Real-Time Prediction Function --------------------
def predict_engagement_change(live_stream_buffer, cnn_classifier, lstm_model, sequence_length=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Get the latest sequence from the buffer
    latest_sequence = list(live_stream_buffer)
    if len(latest_sequence) < sequence_length:
        return None, "Not enough data yet."

    # 2. Convert to tensor and pass to LSTM
    input_tensor = torch.stack(latest_sequence[-sequence_length:]).unsqueeze(0).to(device)
    
    # 3. Get the prediction
    lstm_model.eval()
    with torch.no_grad():
        prediction = lstm_model(input_tensor)
        # The output is [1, seq_len, 3], but we only care about the final prediction
        predicted_scores = prediction[0, -1, :].cpu().numpy()

    # 4. Analyze the prediction for a potential change
    current_engagement = np.argmax(predicted_scores) # The predicted class for the current moment
    # You can implement logic here to analyze predicted_scores for a significant change
    # e.g., if the probability of 'barely engaged' jumps significantly.
    
    return predicted_scores, "Potential change detected!" # Or a more specific message