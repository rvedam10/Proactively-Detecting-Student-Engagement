import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import mediapipe as mp
import collections

# -------------------- MediaPipe Setup (for efficiency) --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

# -------------------- Utility Functions (unchanged) --------------------
# segment_frame, extract_keyframes, get_transform, get_headpose_gaze_features
# These remain the same as in the previous version.

# -------------------- CNN + headpose feature extraction --------------------
def extract_features_from_segment(frame, cnn_model, transform):
    input_tensor = transform(frame).unsqueeze(0)
    cnn_feat = cnn_model(input_tensor).squeeze(0)
    hg_feat = get_headpose_gaze_features(frame)
    hg_feat = torch.tensor(hg_feat, dtype=torch.float32)
    feature_vector = torch.cat([cnn_feat, hg_feat], dim=0)
    return feature_vector

# -------------------- Per-Frame Engagement Classifier (CNN) --------------------
# We will use this to generate the per-frame scores for the LSTM.
class EngagementCNN(nn.Module):
    def __init__(self, input_dim=519, num_classes=3):
        super(EngagementCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x # Return logits before softmax

# -------------------- Sequence Prediction Model (LSTM) --------------------
class EngagementLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=2, output_dim=3):
        super(EngagementLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # We need to predict a sequence of outputs, so we don't just take the last timestep
        out = self.fc(lstm_out)
        return self.softmax(out)

# -------------------- Training Data Preparation --------------------
def create_training_sequences(video_scores, sequence_length=10, prediction_length=2):
    X, y = [], []
    for scores in video_scores:
        scores = scores.squeeze(0) # Remove the batch dimension
        for i in range(len(scores) - sequence_length - prediction_length):
            # Input is a sequence of past scores
            input_seq = scores[i:i + sequence_length]
            # Target is the next sequence of scores
            target_seq = scores[i + sequence_length:i + sequence_length + prediction_length]
            X.append(input_seq)
            y.append(target_seq)
    return X, y

# -------------------- Full Training Loop --------------------
def train_model_for_prediction(video_paths, labels, predictor, cnn_feature_extractor, num_epochs=10, batch_size=4, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_classifier = EngagementCNN(num_classes=3).to(device)
    lstm_model = EngagementLSTM(input_dim=3, output_dim=3).to(device) # LSTM now predicts 3 scores

    params = list(cnn_classifier.parameters()) + list(lstm_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss() # Use MSELoss for regression/prediction

    cnn_classifier.train()
    lstm_model.train()
    
    # First, process all videos to get their sequences of per-frame scores
    all_video_scores = []
    for video_path in video_paths:
        # Placeholder for real video processing
        # This part should be replaced with your actual process_video function
        # from the previous version, but adapted to return the sequence of scores.
        # For this example, let's assume we have a way to get the scores.
        pass # The real implementation would be here

    # Then, create sequences for training
    X_train, y_train = create_training_sequences(all_video_scores)

    # Convert to tensors
    X_train = torch.stack(X_train).to(device)
    y_train = torch.stack(y_train).to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(X_train):.4f}")
    return cnn_classifier, lstm_model

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