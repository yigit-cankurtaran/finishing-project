import numpy as np
import matplotlib.pyplot as plt
from on_actual_data import predict_signal, SignalCNN
import wfdb
import torch


def visualize_ecg_predictions(record_path, model, window_size=250):
    """
    Visualize ECG data and model predictions
    Args:
        record_path: Path to the MITBIH record
        model: Trained PyTorch model
        window_size: Size of the sliding window
    """
    # Load the record and annotations
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, "atr")

    # Get signal data (first lead)
    signal = record.p_signal.T[0]

    # Get predictions from model
    predictions = predict_signal(model, signal)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot original signal with annotations
    time = np.arange(len(signal)) / record.fs  # Convert to seconds
    ax1.plot(time, signal, "b-", label="ECG Signal", alpha=0.6)

    # Plot ground truth annotations
    abnormal_types = ["V", "A", "L", "R", "F", "f", "j", "a", "S", "E", "J"]
    for sample, symbol in zip(annotations.sample, annotations.symbol):
        if sample < len(signal) and symbol in abnormal_types:
            ax1.axvline(x=sample / record.fs, color="r", alpha=0.3)

    # Plot predictions
    # Resize predictions to match signal length
    pred_time = np.linspace(0, len(signal) / record.fs, len(predictions))
    ax2.plot(
        pred_time, predictions.squeeze(), "g-", label="Model Confidence", alpha=0.7
    )
    ax2.axhline(y=0.5, color="r", linestyle="--", label="Detection Threshold")

    # Highlight regions where model predicts abnormality
    predictions_resized = np.interp(time, pred_time, predictions.squeeze())
    abnormal_regions = predictions_resized > 0.5
    ax1.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=abnormal_regions,
        color="yellow",
        alpha=0.2,
        label="Model-detected Abnormality",
    )

    # Customize plots
    ax1.set_title(
        "ECG Signal (Blue) with Ground Truth Annotations (Red) and Model Predictions (Yellow)"
    )
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Model Prediction Confidence")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Abnormality Confidence")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # device setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    # Load a specific record (e.g., record 100)
    record_path = "data/100"
    window_size = 250  # Define window size

    # Create model instance with input_shape instead of window_size
    model = SignalCNN(
        input_shape=window_size
    )  # Changed from window_size=250 to input_shape=250
    model.to(device)
    # Load the saved state dictionary
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Visualize the data and predictions
    visualize_ecg_predictions(record_path, model)
