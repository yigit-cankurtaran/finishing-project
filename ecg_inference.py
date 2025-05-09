import os
import sys
import numpy as np
import torch
import wfdb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from ecg_training import EnhancedSignalCNN, load_scp_statements, interpret_predictions
from enhanced_ecg_train import ECGNet
import device_func


def load_ecg_file(record_path):
    """
    Load an ECG recording from .dat and .hea files.

    Args:
        record_path: Path to the record without file extension

    Returns:
        signal: The ECG signal data
    """
    try:
        # Load the record using wfdb
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal

        # Print record information
        print(f"Loaded ECG record: {record_path}")
        print(f"Sampling frequency: {record.fs} Hz")
        print(f"Number of channels: {signal.shape[1]}")
        print(
            f"Signal length: {signal.shape[0]} samples ({signal.shape[0] / record.fs:.2f} seconds)"
        )

        return signal, record.fs
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return None, None


def preprocess_signal(signal, target_length=5000):
    """
    Preprocess the ECG signal for model input.

    Args:
        signal: The raw ECG signal
        target_length: The required length for the model input

    Returns:
        processed_signal: The preprocessed signal ready for model input
    """
    # Standardize each channel
    scaler = StandardScaler()
    processed_signal = scaler.fit_transform(signal)

    # Handle signal length
    if processed_signal.shape[0] > target_length:
        # If signal is too long, take the middle segment
        start = (processed_signal.shape[0] - target_length) // 2
        processed_signal = processed_signal[start : start + target_length]
    elif processed_signal.shape[0] < target_length:
        # If signal is too short, pad with zeros
        pad_width = ((0, target_length - processed_signal.shape[0]), (0, 0))
        processed_signal = np.pad(processed_signal, pad_width, mode="constant")

    return processed_signal


def plot_ecg(signal, fs, predictions=None, save_path=None):
    """
    Plot the ECG signal with predictions.

    Args:
        signal: The ECG signal data
        fs: Sampling frequency
        predictions: Model predictions (optional)
        save_path: Path to save the plot (optional)
    """
    # Calculate time axis in seconds
    time = np.arange(signal.shape[0]) / fs

    # Create a figure with subplots for each channel
    n_channels = signal.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)

    # Plot each channel
    for i in range(n_channels):
        if n_channels == 1:
            ax = axes
        else:
            ax = axes[i]

        ax.plot(time, signal[:, i])
        ax.set_ylabel(f"Channel {i + 1}")
        ax.grid(True)

    # Set common labels
    axes[-1].set_xlabel("Time (s)")

    # Add predictions as title if available
    if predictions:
        pred_str = " | ".join([f"{cls}: {prob:.2f}" for cls, prob in predictions[:3]])
        fig.suptitle(f"ECG with Predictions: {pred_str}", fontsize=14)
    else:
        fig.suptitle("ECG Signal", fontsize=14)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def load_model(model_path, scp_statements_csv, window_size=5000):
    """
    Load the trained model and diagnostic class mappings.

    Args:
        model_path: Path to the saved model
        scp_statements_csv: Path to the SCP statements CSV file
        window_size: Input window size for the model

    Returns:
        model: The loaded model
        diagnostic_classes: List of diagnostic classes
        class_to_index: Mapping from class names to indices
        index_to_class: Mapping from indices to class names
    """
    # Get device
    device = device_func.device_func()

    # Load SCP statements and mappings
    scp_to_diagnostic_class, class_to_index, index_to_class, diagnostic_classes = (
        load_scp_statements(scp_statements_csv)
    )
    n_classes = len(diagnostic_classes)

    # Initialize model
    model = ECGNet(n_ch=12, n_cls=n_classes).to(device)

    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)

    return model, diagnostic_classes, class_to_index, index_to_class


def run_inference(model, signal, diagnostic_classes, class_to_index, threshold=0.5):
    """
    Run inference on an ECG signal.

    Args:
        model: The trained model
        signal: The preprocessed ECG signal
        diagnostic_classes: List of diagnostic classes
        class_to_index: Mapping from class names to indices
        threshold: Probability threshold for positive predictions

    Returns:
        predictions: List of (class, probability) tuples
    """
    device = next(model.parameters()).device

    # Convert to tensor and add batch dimension
    signal_tensor = (
        torch.tensor(signal, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(signal_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    # Get predicted classes (probability > threshold)
    predicted_indices = np.where(probabilities > threshold)[0]

    # Map indices to diagnostic classes
    predictions = []
    for idx in predicted_indices:
        diagnostic_class = diagnostic_classes[idx]
        prob = probabilities[idx]
        predictions.append((diagnostic_class, prob))

    # Sort by probability
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on ECG data")
    parser.add_argument(
        "--record",
        type=str,
        required=True,
        help="Path to the ECG record (without extension)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ecgnet_best.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--scp_csv",
        type=str,
        default="./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv",
        help="Path to the SCP statements CSV file",
    )
    parser.add_argument(
        "--window_size", type=int, default=5000, help="Window size for the model input"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive predictions",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the ECG signal with predictions"
    )
    parser.add_argument(
        "--save_plot", type=str, default=None, help="Path to save the plot"
    )

    args = parser.parse_args()

    # Load the model
    model, diagnostic_classes, class_to_index, index_to_class = load_model(
        args.model, args.scp_csv, args.window_size
    )

    # Load the ECG record
    signal, fs = load_ecg_file(args.record)
    if signal is None:
        print("Failed to load ECG record")
        return

    # Preprocess the signal
    processed_signal = preprocess_signal(signal, args.window_size)

    # Run inference
    predictions = run_inference(
        model, processed_signal, diagnostic_classes, class_to_index, args.threshold
    )

    # Print predictions
    print("\nPredictions:")
    for i, (diagnostic_class, probability) in enumerate(predictions):
        print(f"{i + 1}. {diagnostic_class}: {probability:.4f}")

    # Plot if requested
    if args.plot or args.save_plot:
        plot_ecg(signal, fs, predictions, args.save_plot)


if __name__ == "__main__":
    main()
