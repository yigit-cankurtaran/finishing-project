import wfdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import device_func
import ast  # safer eval
import os
import matplotlib.pyplot as plt
from collections import Counter


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


# Enhanced CNN for multi-label classification
class EnhancedSignalCNN(nn.Module):
    def __init__(self, input_shape, n_channels=12, n_classes=5):
        super(EnhancedSignalCNN, self).__init__()
        print(
            f"Initializing CNN with input shape: {input_shape}, channels: {n_channels}, classes: {n_classes}"
        )

        # Convolutional layers
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Calculate output shape dynamically
        with torch.no_grad():
            x = torch.randn(1, n_channels, input_shape)
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))
            x = x.flatten(1)
            flat_features = x.shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        if len(x.shape) == 2:  # if input is [batch, window_size]
            x = x.unsqueeze(1)  # add channel dimension
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Load and process SCP statements
def load_scp_statements(scp_csv):
    """Load SCP statements and create a mapping of diagnostic classes."""
    scp_df = pd.read_csv(scp_csv)

    # Filter for diagnostic statements (diagnostic=1.0)
    diagnostic_df = scp_df[scp_df["diagnostic"] == 1.0]

    # Create a mapping of SCP codes to diagnostic classes
    scp_to_diagnostic_class = {}
    for _, row in diagnostic_df.iterrows():
        if pd.notna(row["diagnostic_class"]):
            scp_to_diagnostic_class[row.iloc[0]] = row["diagnostic_class"]

    # Get unique diagnostic classes
    diagnostic_classes = sorted(list(set(scp_to_diagnostic_class.values())))

    # Create mapping dictionaries
    class_to_index = {cls: idx for idx, cls in enumerate(diagnostic_classes)}
    index_to_class = {idx: cls for idx, cls in enumerate(diagnostic_classes)}

    return scp_to_diagnostic_class, class_to_index, index_to_class, diagnostic_classes


# Enhanced function to load PTB-XL data with multi-label support
def load_ptbxl_data(
    ptbxl_csv, signal_dir, scp_statements_csv, sampling_rate=500, window_size=5000
):
    """
    Load PTB-XL data with multi-label classification support.
    Returns data and multi-hot encoded labels for diagnostic classes.
    """
    # Load SCP statements and mappings
    scp_to_diagnostic_class, class_to_index, index_to_class, diagnostic_classes = (
        load_scp_statements(scp_statements_csv)
    )
    n_classes = len(diagnostic_classes)
    print(f"Found {n_classes} diagnostic classes: {diagnostic_classes}")

    # Load PTB-XL database
    df = pd.read_csv(ptbxl_csv)

    # Select high-resolution signals if sampling rate is 500
    if sampling_rate == 500:
        df = df[df["filename_hr"].notna()]
        file_column = "filename_hr"
    elif sampling_rate == 100:
        df = df[df["filename_lr"].notna()]
        file_column = "filename_lr"
    else:
        raise ValueError("Invalid sampling rate. Choose 500 or 100 Hz.")

    data, labels = [], []
    label_counts = Counter()

    for _, row in df.iterrows():
        record_path = f"{signal_dir}/{row[file_column]}"
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal  # shape: [timesteps, 12]
        except Exception as e:
            print(f"Error loading record {record_path}: {e}")
            continue

        # Safely parse scp_codes
        try:
            scp_codes = ast.literal_eval(
                row["scp_codes"]
            )  # safely convert string to dict
        except (ValueError, SyntaxError):
            print(f"Invalid scp_codes: {row['scp_codes']}")
            continue

        # Create multi-hot encoded label vector
        multi_hot_label = np.zeros(n_classes)

        # Map SCP codes to diagnostic classes
        for scp_code in scp_codes.keys():
            if scp_code in scp_to_diagnostic_class:
                diagnostic_class = scp_to_diagnostic_class[scp_code]
                if diagnostic_class in class_to_index:
                    class_idx = class_to_index[diagnostic_class]
                    multi_hot_label[class_idx] = 1
                    label_counts[diagnostic_class] += 1

        # Skip records with no valid diagnostic classes
        if np.sum(multi_hot_label) == 0:
            continue

        # Segment the signal
        for i in range(0, signal.shape[0] - window_size + 1, window_size // 2):
            segment = signal[i : i + window_size]
            data.append(segment)
            labels.append(multi_hot_label)

    print(f"Label distribution: {label_counts}")
    return (
        np.array(data),
        np.array(labels),
        diagnostic_classes,
        class_to_index,
        index_to_class,
    )


# Function to visualize label distribution
def plot_label_distribution(labels, class_names):
    """Plot the distribution of labels in the dataset."""
    label_counts = np.sum(labels, axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, label_counts)
    plt.xticks(rotation=90)
    plt.title("Label Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.close()


# Enhanced training function
def train_and_evaluate(
    ptbxl_csv,
    signal_dir,
    scp_statements_csv,
    model_path="enhanced_model.pth",
    window_size=5000,
):
    device = device_func.device_func()
    print(f"Training on {device}")

    # Load data with multi-label support
    X, y, diagnostic_classes, class_to_index, index_to_class = load_ptbxl_data(
        ptbxl_csv, signal_dir, scp_statements_csv, window_size=window_size
    )

    n_classes = len(diagnostic_classes)

    # Plot label distribution
    plot_label_distribution(y, diagnostic_classes)

    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = EnhancedSignalCNN(
            input_shape=window_size, n_channels=12, n_classes=n_classes
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, diagnostic_classes, class_to_index, index_to_class

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.shape[0] == y.shape[1] else None,
    )

    # Add shape checks
    print(
        f"Before scaling - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
    )

    # Scale data
    scaler = StandardScaler()
    X_train = np.array([scaler.fit_transform(x) for x in X_train])
    X_test = np.array([scaler.transform(x) for x in X_test])

    print(f"After scaling - X_train shape: {X_train.shape}")

    # Create datasets and dataloaders
    train_dataset = SignalDataset(X_train, y_train)
    test_dataset = SignalDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Setup enhanced CNN
    model = EnhancedSignalCNN(
        input_shape=window_size, n_channels=12, n_classes=n_classes
    ).to(device)

    # Binary cross entropy loss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Training loop with model saving and metrics calculation
    epochs = 30
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}

    # Early stopping parameters
    patience = 5
    patience_counter = 0

    print("\nStarting training with early stopping (patience=5)...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Convert outputs to predictions (0 or 1)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Calculate accuracy (exact matches)
                batch_correct = torch.all(preds == labels, dim=1).sum().item()
                correct_predictions += batch_correct
                total_samples += labels.size(0)

                # Calculate per-class accuracy
                for i in range(n_classes):
                    class_preds = preds[:, i]
                    class_labels = labels[:, i]
                    class_correct = (class_preds == class_labels).sum().item()
                    class_total = class_labels.size(0)

        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Calculate exact match accuracy (percentage)
        accuracy = (correct_predictions / total_samples) * 100
        history["val_accuracy"].append(accuracy)

        # Calculate hamming accuracy (percentage)
        hamming_accuracy = np.mean(all_preds == all_labels) * 100

        # Calculate F1 score for each class and average
        f1_scores = []
        for i in range(n_classes):
            if (
                np.sum(all_labels[:, i]) > 0
            ):  # Only calculate if class exists in test set
                f1 = f1_score(all_labels[:, i], all_preds[:, i])
                f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_val_loss = val_loss / len(test_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(avg_f1)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"F1: {avg_f1:.4f}, Exact Match Accuracy: {accuracy:.2f}%, Hamming Accuracy: {hamming_accuracy:.2f}%"
        )

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save model if it has the best validation loss so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            print(f"Saving best model with validation loss: {best_loss:.4f}")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

            # Check if early stopping criteria is met
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history["val_f1"], label="F1 Score")
    plt.legend()
    plt.title("F1 Score")

    plt.subplot(1, 3, 3)
    plt.plot(history["val_accuracy"], label="Accuracy")
    plt.legend()
    plt.title("Exact Match Accuracy (%)")

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

    print("\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")

    return model, diagnostic_classes, class_to_index, index_to_class


# Function to interpret model predictions
def interpret_predictions(model, signal, diagnostic_classes, class_to_index, device):
    """
    Interpret model predictions for a given ECG signal.
    Returns the predicted diagnostic classes and their probabilities.
    """
    model.eval()

    # Preprocess signal
    scaler = StandardScaler()
    scaled_signal = scaler.fit_transform(signal)

    # Convert to tensor and add batch dimension
    signal_tensor = (
        torch.tensor(scaled_signal, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(signal_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    # Get predicted classes (probability > 0.5)
    predicted_indices = np.where(probabilities > 0.5)[0]

    # Map indices to diagnostic classes
    predictions = []
    for idx in predicted_indices:
        diagnostic_class = diagnostic_classes[idx]
        prob = probabilities[idx]
        predictions.append((diagnostic_class, prob))

    # Sort by probability
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions


if __name__ == "__main__":
    ptbxl_csv = "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    signal_dir = "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/signals"
    scp_statements_csv = "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    model_path = "enhanced_model.pth"

    # Train and evaluate the enhanced model
    model, diagnostic_classes, class_to_index, index_to_class = train_and_evaluate(
        ptbxl_csv, signal_dir, scp_statements_csv, model_path
    )

    print(
        "\nModel training complete. The model can now predict the following diagnostic classes:"
    )
    for i, cls in enumerate(diagnostic_classes):
        print(f"{i + 1}. {cls}")

    print(
        "\nYou can now use this model to interpret ECG signals and diagnose heart conditions."
    )
