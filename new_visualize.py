import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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


def visualize_training_history(train_accuracies, val_accuracies, save_path=None):
    """
    Visualize training and validation accuracy over epochs.

    Args:
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, "b-", label="Training Accuracy")
    plt.plot(epochs, val_accuracies, "r-", label="Validation Accuracy")

    plt.title("Model Accuracy over Training Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Train model and visualize training history
    model, train_accuracies, val_accuracies = train_signal_model()
    visualize_training_history(train_accuracies, val_accuracies)

    # Visualize random test cases
    visualize_random_test_cases(num_samples=4)
