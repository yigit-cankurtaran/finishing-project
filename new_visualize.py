import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
from on_actual_data import SignalCNN, preprocess_signal_data
import pandas as pd


def load_trained_model(window_size=250, model_path="best_model.pth", device="cpu"):
    """
    Load the trained model for inference.
    """
    model = SignalCNN(input_shape=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_abnormal_regions(model, signal_data, window_size=250, threshold=0.5):
    """
    Generate abnormal regions based on model predictions.

    Args:
        model: Trained model
        signal_data: Array of signal data
        window_size: Size of each segment to analyze
        threshold: Threshold for abnormality

    Returns:
        List of dicts with start_idx, end_idx, and confidence for abnormal regions
    """
    segments, _ = preprocess_signal_data(
        signal_data, np.zeros(len(signal_data)), window_size=window_size
    )
    segments = torch.tensor(
        segments, dtype=torch.float32
    )  # shape: [batch_size, window_size]
    segments = segments.unsqueeze(
        1
    )  # Add channel dimension, shape: [batch_size, 1, window_size]
    device = next(model.parameters()).device
    segments = segments.to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(segments)).squeeze()
        abnormal_regions = []
        for i, confidence in enumerate(outputs):
            if confidence > threshold:
                start_idx = i * (window_size // 2)
                end_idx = start_idx + window_size
                abnormal_regions.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "confidence": confidence.item(),
                    }
                )
    return abnormal_regions


def visualize_predictions(
    signal_data, abnormal_regions, window_size=250, save_path=None
):
    """
    Visualize the ECG signal with highlighted abnormal regions.

    Args:
        signal_data: Original signal data array
        abnormal_regions: List of dicts with start_idx, end_idx, and confidence
        window_size: Size of the window used for prediction
        save_path: Optional path to save the plot
    """
    # Create figure and axis
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Plot original signal
    time_points = np.arange(len(signal_data))
    plt.plot(time_points, signal_data, "b-", label="ECG Signal", linewidth=1)

    # Create colormap for confidence scores
    max_confidence = (
        max([region["confidence"] for region in abnormal_regions])
        if abnormal_regions
        else 1.0
    )

    # Plot abnormal regions
    for region in abnormal_regions:
        start = region["start_idx"]
        end = region["end_idx"]
        confidence = region["confidence"]

        # Calculate alpha based on confidence
        alpha = 0.3 * (confidence / max_confidence)

        # Add highlighted rectangle
        rect = Rectangle(
            (start, plt.ylim()[0]),
            end - start,
            plt.ylim()[1] - plt.ylim()[0],
            facecolor="red",
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Add confidence score text
        plt.text(
            start, plt.ylim()[1] * 0.9, f"{confidence:.2f}", fontsize=8, rotation=90
        )

    # Customize plot
    plt.title("ECG Signal Analysis with Abnormality Detection")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # Add legend
    normal_patch = plt.Rectangle((0, 0), 1, 1, fc="b", alpha=0.5, label="Normal Signal")
    abnormal_patch = plt.Rectangle(
        (0, 0), 1, 1, fc="red", alpha=0.3, label="Abnormal Region"
    )
    plt.legend(handles=[normal_patch, abnormal_patch])

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Load signal data (replace with actual test data path)
    test_df = pd.read_csv("data/mitbih_test.csv")
    signal_data = test_df.values[:, :-1].flatten()  # Flatten the data for full signal
    window_size = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(window_size=window_size, device=device)

    # Generate abnormal regions
    abnormal_regions = generate_abnormal_regions(
        model, signal_data, window_size=window_size
    )

    # Visualize predictions
    visualize_predictions(signal_data, abnormal_regions, window_size=window_size)
