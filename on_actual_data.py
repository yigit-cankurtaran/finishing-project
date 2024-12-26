import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wfdb  # waveform database
import device_func


# custom dataset class to handle ecg data
# converts numpy arrays to pytorch tensors for training
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
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

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
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# loads ecg data from mit-bih database
def load_mitbih_data(record_paths):
    all_signals = []
    all_labels = []

    for path in record_paths:
        # load signal data
        record = wfdb.rdrecord(path)
        signals = record.p_signal.T[0]  # get first lead only

        # load annotations (labels)
        annotations = wfdb.rdann(path, "atr")

        # create binary labels array
        labels = np.zeros(len(signals))
        for sample in annotations.sample:
            if sample < len(signals):
                labels[sample] = 1

        all_signals.append(signals)
        all_labels.append(labels)

    return np.concatenate(all_signals), np.concatenate(all_labels)


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


# trains the model on preprocessed data
def train_signal_model(X_data, y_labels, window_size=250, sampling_rate=360):
    device = device_func.device_func()
    print(f"Training on {device}")

    # Ensure inputs are properly scaled
    X_data = X_data.astype(
        np.float32
    )  # in case of bad data change this to torch.float32
    y_labels = y_labels.astype(np.float32)

    # preprocess data
    X_processed, y_processed = preprocess_signal_data(
        X_data, y_labels, sampling_rate, window_size
    )

    # Add checks for NaN values
    if np.isnan(X_processed).any():
        print("Warning: NaN values in processed data")
        X_processed = np.nan_to_num(X_processed)

    # split into train/val/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # create data loaders
    train_dataset = SignalDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = SignalDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Modified model initialization and loss setup
    model = SignalCNN(window_size).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use logits-based loss
    optimizer = optim.Adam(
        model.parameters(), lr=0.0003, weight_decay=1e-5
    )  # less learning rate + weight decay
    # preventing overfiting

    # Add scheduler definition here
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # training settings
    epochs = 50
    best_val_acc = 0
    patience = 5  # early stopping
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # try to load existing model if it exists
    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded existing model, continuing training...")
    except FileNotFoundError:
        print("Training new model...")

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            # Print gradient info after backward pass
            if epoch == 0 and batch_idx == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(
                            f"{name} grad range: {param.grad.min():.5f} to {param.grad.max():.5f}"
                        )

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # calculate metrics
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs).data > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs).data > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")  # pth = pytorch extension
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"early stopping triggered at epoch {epoch+1}")
            break

        # save history and update scheduler
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

    # after training, evaluate on test set
    test_dataset = SignalDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16)
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs).data > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted.squeeze() == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, history


# makes predictions on new data
def predict_signal(model, new_data, window_size=250, sampling_rate=360):
    device = device_func.device_func()
    print(f"Predicting on {device}")
    model = model.to(device)
    model.eval()

    # preprocess new data
    dummy_labels = np.zeros(len(new_data))
    processed_data, _ = preprocess_signal_data(
        new_data, dummy_labels, sampling_rate, window_size
    )

    # run predictions in batches
    predictions = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(processed_data), batch_size):
            batch = torch.FloatTensor(processed_data[i : i + batch_size]).to(device)
            batch_preds = torch.sigmoid(model(batch))  # Apply sigmoid here
            predictions.append(batch_preds.cpu().numpy())

    return np.concatenate(predictions)


# main execution block
if __name__ == "__main__":
    # check if data directory exists
    import os

    data_dir = "data"

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Downloading MIT-BIH database...")
        wfdb.dl_database("mitdb", dl_dir="data")
    else:
        print("Found existing MIT-BIH database")

    # Actual MIT-BIH record numbers
    record_numbers = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        121,
        122,
        123,
        124,
        200,
        201,
        202,
        203,
        205,
        207,
        208,
        209,
        210,
        212,
        213,
        214,
        215,
        217,
        219,
        220,
        221,
        222,
        223,
        228,
        230,
        231,
        232,
        233,
        234,
    ]

    # Get record paths
    record_paths = [
        f"data/{str(i)}" for i in record_numbers if os.path.exists(f"data/{str(i)}.dat")
    ]

    print(f"Found {len(record_paths)} records")
    # load and process data
    X_data, y_labels = load_mitbih_data(record_paths)

    # Convert to float32
    X_data = X_data.astype(np.float32)
    y_labels = y_labels.astype(np.float32)

    print(f"Loaded data shape: {X_data.shape}")
    print(f"Labels shape: {y_labels.shape}")

    # train model
    model, history = train_signal_model(X_data, y_labels)

    # test on new data (use first unseen record)
    test_record_num = next(i for i in record_numbers if f"data/{i}" not in record_paths)
    test_record = wfdb.rdrecord(f"data/{test_record_num}")
    test_signal = test_record.p_signal.T[0].astype(np.float32)  # Convert to float32
    predictions = predict_signal(model, test_signal)
    print(f"Predictions shape: {predictions.shape}")
