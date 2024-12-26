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

    # Get predictions and print range for debugging
    predictions = predict_signal(model, signal)
    print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"Number of abnormal predictions: {(predictions > 0.5).sum()}")

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot original signal
    time = np.arange(len(signal)) / record.fs
    ax.plot(time, signal, "b-", label="ECG Signal", alpha=0.6)

    # Plot ground truth annotations in red
    abnormal_types = ["V", "A", "L", "R", "F", "f", "j", "a", "S", "E", "J"]
    for sample, symbol in zip(annotations.sample, annotations.symbol):
        if sample < len(signal) and symbol in abnormal_types:
            ax.axvline(
                x=sample / record.fs,
                color="red",
                alpha=0.4,
                label="Ground Truth" if sample == annotations.sample[0] else "",
            )

    # Plot model predictions in green with higher visibility
    pred_time = np.linspace(0, len(signal) / record.fs, len(predictions))
    predictions_resized = np.interp(time, pred_time, predictions.squeeze())
    abnormal_regions = predictions_resized > 0.5

    # Make predictions more visible
    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=abnormal_regions,
        color="lime",
        alpha=0.4,
        label="Model Predictions",
    )

    ax.set_title("ECG Signal Analysis")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

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

    # Verify model loaded correctly
    print("Model architecture:", model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Test model with random input
    test_input = torch.randn(1, window_size).to(device)
    with torch.no_grad():
        test_output = model(test_input)
        print(
            f"Test output range: {test_output.min().item():.4f} to {test_output.max().item():.4f}"
        )

    # Visualize the data and predictions
    visualize_ecg_predictions(record_path, model)
