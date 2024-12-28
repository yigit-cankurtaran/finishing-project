import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import device_func


class SignalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# neural network model for ecg analysis
# uses 1d convolutions since ecg data is a time series
class SignalCNN(nn.Module):
    def __init__(self, input_shape):
        super(SignalCNN, self).__init__()
        print(f"Initializing CNN with input shape: {input_shape}")

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Calculate output shape
        with torch.no_grad():
            x = torch.randn(1, 1, input_shape)
            print(f"Initial x shape: {x.shape}")

            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            print(f"After conv1: {x.shape}")

            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            print(f"After conv2: {x.shape}")

            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            print(f"After conv3: {x.shape}")

            x = x.flatten(1)
            flat_features = x.shape[1]

        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# splits signal into overlapping windows and normalizes
def preprocess_signal_data(data, labels, sampling_rate=360, window_size=250):
    segments = []
    segment_labels = []

    # create windows with 50% overlap
    for i in range(0, len(data) - window_size, window_size // 2):
        segment = data[i : i + window_size]
        if len(segment) == window_size:
            segments.append(segment)
            # label window based on majority
            segment_label = np.mean(labels[i : i + window_size])
            segment_labels.append(1 if segment_label >= 0.5 else 0)

    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    # normalize each segment independently
    scaler = StandardScaler()
    segments_scaled = np.array(
        [scaler.fit_transform(segment.reshape(-1, 1)).flatten() for segment in segments]
    )

    return segments_scaled, segment_labels


def train_signal_model(X_data, y_labels, window_size=250):
    device = device_func.device_func()
    print(f"Training on {device}")

    # Split into train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # Create data loaders with larger batch size for bigger dataset
    train_dataset = SignalDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = SignalDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SignalCNN(window_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True
    )

    epochs = 30
    best_val_acc = 0
    patience = 7
    patience_counter = 0

    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded existing model")
    except FileNotFoundError:
        print("Training new model")

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.squeeze() == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        scheduler.step(val_loss)

    model.load_state_dict(torch.load("best_model.pth"))
    return model, train_accuracies, val_accuracies


def train_and_evaluate():
    device = device_func.device_func()
    print(f"Training on {device}")

    # Load MITBIH datasets
    train_df = pd.read_csv("data/mitbih_train.csv")
    test_df = pd.read_csv("data/mitbih_test.csv")

    # Prepare data
    X_train = train_df.values[:, :-1].astype(np.float32)
    y_train = train_df.values[:, -1].astype(np.float32)
    X_test = test_df.values[:, :-1].astype(np.float32)
    y_test = test_df.values[:, -1].astype(np.float32)

    # Create data loaders
    train_dataset = SignalDataset(X_train, y_train)
    test_dataset = SignalDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model and training components
    model = SignalCNN(input_shape=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Training loop
    epochs = 30
    best_test_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels).sum().item()

        # Testing
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted.squeeze() == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step(1 - test_acc / 100)

    return model


def predict_signal(
    signal_data, model_path="best_model.pth", window_size=250, threshold=0.5
):
    """
    Predict abnormalities in ECG signal using sliding windows.
    Returns indices of abnormal regions with confidence scores.
    """
    device = device_func.device_func()

    # Load and prepare model
    model = SignalCNN(window_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Normalize signal
    scaler = StandardScaler()
    signal_data = scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()

    # Analyze signal with sliding windows
    abnormal_regions = []
    stride = window_size // 2

    with torch.no_grad():
        for i in range(0, len(signal_data) - window_size + 1, stride):
            window = signal_data[i : i + window_size]
            window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
            output = model(window_tensor.unsqueeze(0))
            probability = torch.sigmoid(output).item()

            if probability > threshold:
                abnormal_regions.append(
                    {
                        "start_idx": i,
                        "end_idx": i + window_size,
                        "confidence": probability,
                    }
                )

    return abnormal_regions


if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv("data/mitbih_test.csv")
    signal = test_df.values[:, :-1].astype(np.float32)[0]  # First signal

    # Train model if needed
    model = train_and_evaluate()

    # Test prediction
    abnormal_regions = predict_signal(signal)
    print("\nAbnormal Regions Detected:")
    for region in abnormal_regions:
        print(f"Region from {region['start_idx']} to {region['end_idx']}")
        print(f"Confidence: {region['confidence']:.2f}")
