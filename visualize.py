import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from on_actual_data import predict_signal
import device_func


def visualize_predictions(signal_data, predictions, title="ECG Signal Analysis"):
    """
    Visualize ECG signal and model predictions
    Args:
        signal_data: Raw ECG signal array
        predictions: List of dictionaries containing prediction regions
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot original signal
    time = np.arange(len(signal_data))
    ax.plot(time, signal_data, "b-", label="ECG Signal", alpha=0.6)

    # Plot predicted abnormal regions
    for region in predictions:
        ax.axvspan(
            region["start_idx"],
            region["end_idx"],
            color="red",
            alpha=0.3 * region["confidence"],
            label="Abnormal" if region == predictions[0] else "",
        )

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = device_func.device_func()
    print(f"Using device: {device}")

    # Load test data
    test_df = pd.read_csv("data/mitbih_test.csv")

    # Get first few signals for visualization
    num_signals = 5
    for i in range(num_signals):
        signal = test_df.values[i, :-1].astype(np.float32)
        label = test_df.values[i, -1]

        # Get predictions
        predictions = predict_signal(signal)

        # Visualize
        title = f"ECG Signal {i+1} (Ground Truth Label: {int(label)})"
        visualize_predictions(signal, predictions, title)
