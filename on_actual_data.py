import wfdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import device_func
import ast  # safer eval


# dataset class
class SignalDataset(Dataset):
    def __init__(self, data, labels):
        # Transpose data to (batch, channels, time_steps)
        self.data = torch.tensor(data, dtype=torch.float32).transpose(1, 2)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        print(f"Dataset shapes - Data: {self.data.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# cnn class
class SignalCNN(nn.Module):
    def __init__(self, input_shape, n_channels=12):
        super(SignalCNN, self).__init__()
        print(
            f"Initializing CNN with input shape: {input_shape}, channels: {n_channels}"
        )

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # calculate output shape dynamically
        with torch.no_grad():
            x = torch.randn(1, n_channels, input_shape)
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = x.flatten(1)
            flat_features = x.shape[1]

        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        if len(x.shape) == 2:  # if input is [batch, window_size]
            x = x.unsqueeze(1)  # add channel dimension
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# preprocess ptb-xl signals
def load_ptbxl_data(ptbxl_csv, signal_dir, sampling_rate=500, window_size=5000):
    df = pd.read_csv(ptbxl_csv)

    # select high-resolution signals if sampling rate is 500
    if sampling_rate == 500:
        df = df[df["filename_hr"].notna()]
        file_column = "filename_hr"
    elif sampling_rate == 100:
        df = df[df["filename_lr"].notna()]
        file_column = "filename_lr"
    else:
        raise ValueError("Invalid sampling rate. Choose 500 or 100 Hz.")

    data, labels = [], []
    for _, row in df.iterrows():
        print(f"{signal_dir}/{row[file_column]}")
        record_path = f"{signal_dir}/{row[file_column]}"
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # shape: [timesteps, 12]

        # safely parse scp_codes
        try:
            label_dict = ast.literal_eval(
                row["scp_codes"]
            )  # safely convert string to dict
        except (ValueError, SyntaxError):
            print(f"Invalid scp_codes: {row['scp_codes']}")
            continue

        # classify as normal if "NORM" is in the keys
        binary_label = 1 if "NORM" in label_dict.keys() else 0

        # segment the signal
        for i in range(0, signal.shape[0] - window_size + 1, window_size // 2):
            segment = signal[i : i + window_size]
            data.append(segment)
            labels.append(binary_label)

    return np.array(data), np.array(labels)


# train function
def train_and_evaluate(ptbxl_csv, signal_dir, window_size=5000):
    device = device_func.device_func()
    print(f"Training on {device}")

    X, y = load_ptbxl_data(ptbxl_csv, signal_dir, window_size=window_size)

    # Use y directly in train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Add shape checks
    print(f"Before scaling - X_train shape: {X_train.shape}")

    scaler = StandardScaler()
    X_train = np.array([scaler.fit_transform(x) for x in X_train])
    X_test = np.array([scaler.transform(x) for x in X_test])

    print(f"After scaling - X_train shape: {X_train.shape}")

    train_dataset = SignalDataset(X_train, y_train)
    test_dataset = SignalDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # setup cnn
    model = SignalCNN(input_shape=window_size, n_channels=12).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}"
        )

    return model


if __name__ == "__main__":
    ptbxl_csv = "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    signal_dir = "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/signals"
    model = train_and_evaluate(ptbxl_csv, signal_dir)
