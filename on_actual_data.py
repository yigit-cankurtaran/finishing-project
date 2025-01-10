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

        # Calculate output shape dynamically based on input shape
        with torch.no_grad():
            x = torch.randn(1, 1, input_shape)
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
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
        if len(x.shape) == 2:  # If input is [batch, window_size]
            x = x.unsqueeze(
                1
            )  # Add channel dimension to make it [batch, 1, window_size]
        elif len(x.shape) == 3:  # If input is already [batch, channel, window_size]
            pass  # Don't add another dimension

        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Signal preprocessing
def preprocess_signal_data(data, labels, sampling_rate=360, window_size=250):
    """
    Modified preprocessing function to ensure correct output shape.
    """
    segments = []
    segment_labels = []

    # Ensure data is 1D
    data = data.flatten()
    labels = labels.flatten()

    for i in range(0, len(data) - window_size + 1, window_size // 2):
        segment = data[i : i + window_size]
        if len(segment) == window_size:
            segments.append(segment)
            segment_label = np.mean(labels[i : i + window_size])
            segment_labels.append(1 if segment_label >= 0.5 else 0)

    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    # Scale each segment individually
    scaler = StandardScaler()
    segments_scaled = np.array(
        [scaler.fit_transform(segment.reshape(-1, 1)).ravel() for segment in segments]
    )

    return segments_scaled, segment_labels


def train_signal_model(X_data, y_labels, window_size=250):
    device = device_func.device_func()
    print(f"Training on {device}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

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

    epochs = 3
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


def train_and_evaluate(window_size=250):
    device = device_func.device_func()
    print(f"Training on {device}")

    # Load and prepare data
    normal_df = pd.read_csv("data/ptbdb_normal.csv")
    abnormal_df = pd.read_csv("data/ptbdb_abnormal.csv")

    test_size = min(len(normal_df), len(abnormal_df)) // 4

    # Split and combine data
    normal_train = normal_df.iloc[:-test_size]
    normal_test = normal_df.iloc[-test_size:]
    abnormal_train = abnormal_df.iloc[:-test_size]
    abnormal_test = abnormal_df.iloc[-test_size:]

    train_df = pd.concat([normal_train, abnormal_train])
    test_df = pd.concat([normal_test, abnormal_test])

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Preprocess data
    X_train, y_train = preprocess_signal_data(
        train_df.values[:, :-1].astype(np.float32),
        train_df.values[:, -1].astype(np.float32),
        window_size=window_size,
    )
    X_test, y_test = preprocess_signal_data(
        test_df.values[:, :-1].astype(np.float32),
        test_df.values[:, -1].astype(np.float32),
        window_size=window_size,
    )

    # Setup data loaders
    train_dataset = SignalDataset(X_train, y_train)
    test_dataset = SignalDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model setup
    model = SignalCNN(input_shape=window_size).to(device)

    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded existing model for further training")
    except FileNotFoundError:
        print("Starting with new model")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epochs = 30
    best_acc = 0
    patience_counter = 0
    max_patience = 7

    # Training loop
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

        # Test loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted.squeeze() == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
        )
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break

        scheduler.step(test_loss)

    print(f"Best test accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load("best_model.pth"))
    return model


if __name__ == "__main__":
    model = train_and_evaluate()
