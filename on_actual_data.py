import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wfdb  # waveform database


# custom dataset class to handle ecg data
# converts numpy arrays to pytorch tensors for training
class SignalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# neural network model for ecg analysis
# uses 1d convolutions since ecg data is a time series
class SignalCNN(nn.Module):
    def __init__(self, input_shape):
        super(SignalCNN, self).__init__()
        # first conv layer: input -> 8 channels
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3)
        # second conv layer: 8 -> 16 channels
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        # pooling layer to reduce dimensions
        self.pool = nn.MaxPool1d(2)
        # dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # calculate size of flattened features
        with torch.no_grad():
            x = torch.randn(1, 1, input_shape)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.dropout1(x)
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.dropout2(x)
            x = x.flatten(1)
            flat_features = x.shape[1]

        # fully connected layers for final classification
        self.fc1 = nn.Linear(flat_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # add channel dimension for conv1d
        x = x.unsqueeze(1)
        # first conv block
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        # second conv block
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        # flatten and feed through dense layers
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
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
    # set up device (gpu/cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # preprocess data
    X_processed, y_processed = preprocess_signal_data(
        X_data, y_labels, sampling_rate, window_size
    )

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

    # initialize model and training components
    model = SignalCNN(window_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # training settings
    epochs = 50
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # training loop
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            # backward pass
            loss.backward()
            optimizer.step()

            # calculate metrics
            train_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

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
                predicted = (outputs.data > 0.5).float()
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

    # load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, history


# makes predictions on new data
def predict_signal(model, new_data, window_size=250, sampling_rate=360):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # preprocess new data
    dummy_labels = np.zeros(len(new_data))
    processed_data, _ = preprocess_signal_data(
        new_data, dummy_labels, sampling_rate, window_size
    )

    # run predictions
    model.eval()
    processed_tensor = torch.FloatTensor(processed_data).to(device)
    with torch.no_grad():
        predictions = model(processed_tensor)

    return predictions.cpu().numpy()


# main execution block
if __name__ == "__main__":
    # download and prepare data
    wfdb.dl_database("mitdb", dl_dir="data")
    record_paths = ["data/100", "data/101", "data/102"]
    X_data, y_labels = load_mitbih_data(record_paths)

    print(f"loaded data shape: {X_data.shape}")
    print(f"labels shape: {y_labels.shape}")

    # train model
    model, history = train_signal_model(X_data, y_labels)

    # test on new data
    test_record = wfdb.rdrecord("data/103")
    test_signal = test_record.p_signal.T[0]
    predictions = predict_signal(model, test_signal)
    print(f"predictions shape: {predictions.shape}")
