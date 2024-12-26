import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class SignalDataset(Dataset):
    """Custom Dataset for signal data
    Converts numpy arrays to PyTorch tensors and provides iteration capability
    Required for PyTorch's DataLoader"""

    def __init__(self, data, labels):
        # convert numpy arrays to PyTorch tensors
        # tensors are better for training and run on gpus
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        # labels are binary (0/1) values
        # using them as in "normal vs abnormal heartbeat"
        # or "asleep and awake brain activity" etc.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SignalCNN(nn.Module):
    """1D CNN model for signal classification
    Architecture:
    1. Input signal -> Conv1D -> ReLU -> MaxPool -> Dropout
    2. Conv1D -> ReLU -> MaxPool -> Dropout
    3. Flatten -> Dense -> ReLU
    4. Dense -> Sigmoid (for binary classification)"""

    def __init__(self, input_shape):
        super(SignalCNN, self).__init__()
        # first convolutional layer: 1 input channel -> 16 output channels
        # convolutional layers scan through the signal looking for patterns
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3)
        # conv1d is a 1D convolutional layer
        # data is 1d because it's a time series
        # eeg/ecg/etc
        # second convolutional layer: 16 input channels -> 32 output channels
        # halfed to 8 to prevent overfitting
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        # MaxPooling layer with kernel size 2
        # reduces the number of features in the signal
        self.pool = nn.MaxPool1d(2)
        # dropout layers for regularization
        # prevents overfitting
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # turning off a random 50% of the neurons during training
        # raised from 0.3 to 0.5 to prevent overfitting

        # calculate the size of flattened features dynamically
        # this ensures the model works with different input sizes
        with torch.no_grad():
            # no gradient required for input_shape
            x = torch.randn(1, 1, input_shape)
            x = self.pool(torch.relu(self.conv1(x)))
            # relu is a non-linear activation function
            # turns negative values to 0
            # introduces non-linearity, helps complex patterns
            x = self.dropout1(x)
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.dropout2(x)
            x = x.flatten(1)
            # flattening the tensor converts it to a 1D array
            # flattening is useful for feeding into dense layers
            # dense layers connect EVERY input to every output
            flat_features = x.shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 16)
        self.fc2 = nn.Linear(16, 1)
        # reduced to 16 to prevent overfitting
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Add channel dimension for 1D convolution
        x = x.unsqueeze(1)
        # First conv block
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        # Second conv block
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        # Flatten and feed through dense layers
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def preprocess_signal_data(data, labels, sampling_rate, window_size=1000):
    """Preprocess signal data by segmenting it into overlapping windows

    Args:
        data: Raw signal data (1D array)
        labels: Binary labels for each timepoint
        sampling_rate: Number of samples per second (not currently used)
        window_size: Size of each window segment

    Returns:
        segments_scaled: Normalized window segments
        segment_labels: Labels for each window"""

    segments = []
    segment_labels = []

    # Create overlapping windows (50% overlap)
    # take segments of the signal with overlap
    # each window shares 50% of its data with the next window
    # this is to prevent the model from learning the same features
    #
    # ekg sinyalinin periyoduna göre örnek sayılarını tespit etmemiz gerekir
    for i in range(0, len(data) - window_size, window_size // 2):
        segment = data[i : i + window_size]
        if len(segment) == window_size:
            segments.append(segment)
            # Take average of labels in window and threshold at 0.5
            segment_label = np.mean(labels[i : i + window_size])
            segment_labels.append(1 if segment_label >= 0.5 else 0)

    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    # Normalize each segment independently
    scaler = StandardScaler()
    segments_scaled = np.array(
        [scaler.fit_transform(segment.reshape(-1, 1)).flatten() for segment in segments]
    )

    return segments_scaled, segment_labels


def train_signal_model(X_data, y_labels, window_size=1000, sampling_rate=250):
    """Train the CNN model on signal data

    Args:
        X_data: Raw signal data
        y_labels: Binary labels
        window_size: Size of each window segment
        sampling_rate: Number of samples per second

    Returns:
        model: Trained PyTorch model
        history: Dictionary containing training metrics"""

    # Set device (CPU/GPU)
    # if we're on mac we'll use metal acceleration
    # else we check if cuda is available
    # if not, we use cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using metal acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    X_processed, y_processed = preprocess_signal_data(
        X_data, y_labels, sampling_rate, window_size
    )

    # initial train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=0.2,  # 20% of data is used for testing
        random_state=42,  # seed for reproducibility
        shuffle=True,  # shuffle the data before splitting
        stratify=y_processed,  # Added stratification to maintain class distribution
        # we need to maintain the same distribution of classes
        # class distributoin = number of occurences of each class
        # e.g. 30% 1, 70% 0
    )

    # split train test into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,  # 20% of TOTAL data is used for validation
        random_state=42,  # seed for reproducibility
        shuffle=True,  # shuffle the data before splitting
        stratify=y_train,  # Added stratification to maintain class distribution
        # we need to maintain the same distribution of classes
        # class distributoin = number of occurences of each class
        # e.g. 30% 1, 70% 0
    )

    # Print data shapes and class distribution
    print(f"Training data shape: {X_train.shape}")
    print(f"validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    # the 3 lines above might show an error but they work
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"validation labels distribution: {np.bincount(y_val)}")
    print(f"Testing labels distribution: {np.bincount(y_test)}")

    # Create data loaders
    train_dataset = SignalDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # DataLoader is a class that loads data in batches
    # it's used to improve training speed
    val_dataset = SignalDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_dataset = SignalDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize model, loss function, and optimizer
    model = SignalCNN(window_size).to(device)
    # creating the model, pushing it to whatever "device" is
    # metal if mac, cuda if gpu, else cpu
    criterion = nn.BCELoss()
    # binary-cross entropy loss function
    # used for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # adam optimizer
    # lr = learning rate
    # weight decay = prevents overfitting, small penalty for large weights

    print("Model created")
    print(model)

    # Training loop parameters
    epochs = 50
    best_val_acc = 0
    patience = 10  # number of epochs to wait for improvement
    # stops training if no improvement after this many epochs
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Training loop
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            # process input data layer by layer
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass
            # compute gradients by backpropagation
            # updates weights during training
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # for each batch in validation set
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        scheduler.step(val_loss)

    return model, history


def predict_signal(model, new_data, window_size=1000, sampling_rate=250):
    """Make predictions on new signal data

    Args:
        model: Trained PyTorch model
        new_data: Raw signal data to predict
        window_size: Size of each window segment
        sampling_rate: Number of samples per second

    Returns:
        predictions: Model predictions"""

    # checking device code again
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using metal acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess new data
    dummy_labels = np.zeros(len(new_data))
    processed_data, _ = preprocess_signal_data(
        new_data, dummy_labels, sampling_rate, window_size
    )

    # Make predictions
    processed_tensor = torch.FloatTensor(processed_data).to(device)
    with torch.no_grad():
        predictions = model(processed_tensor)

    return predictions.cpu().numpy()


if __name__ == "__main__":
    print("Starting the program...")

    # Create synthetic data with structure
    signal_length = 100000
    t = np.linspace(0, 10, signal_length)

    # baseline wander (things like body movement from breathing etc.)
    low_freq = 0.3
    baseline_wander = 0.2 * np.sin(2 * np.pi * low_freq * t)

    # emg interference (muscle acitvity)
    emg_noise = np.random.normal(0, 0.05, signal_length)

    # electrode noise (noise from the sensor)
    spike_amp = 0.5
    num_spikes = 10
    spike_indices = np.random.choice(signal_length, num_spikes, replace=False)
    spike_signal = np.zeros(signal_length)
    spike_signal[spike_indices] = np.random.normal(0, spike_amp, num_spikes)

    # Generate signal: sine wave + noise
    X_dummy = np.sin(2 * np.pi * t) + baseline_wander + emg_noise + spike_signal
    # Create labels correlated with signal
    y_dummy = (X_dummy > 0).astype(int)

    print(f"Created dummy data with shape: {X_dummy.shape}")
    print(f"Labels shape: {y_dummy.shape}")

    # Train model
    model, history = train_signal_model(X_dummy, y_dummy)

    # Test predictions
    print("\nTesting predictions...")
    new_data = np.random.randn(2000)
    predictions = predict_signal(model, new_data)
    print(f"Predictions shape: {predictions.shape}")

    # Print final metrics
    print("\nTraining History:")
    print("Final training accuracy:", history["train_acc"][-1])
    print("Final validation accuracy:", history["val_acc"][-1])
