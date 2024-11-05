import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# might give a warning but it works, safe to ignore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


# prototype model for the eeg/ecg signals
def preprocess_signal_data(data, labels, sampling_rate, window_size=1000):
    """
    Preprocess EEG/ECG signal data by segmenting it into windows
    Parameters:
    data: numpy array of signal data
    labels: numpy array of labels
    sampling_rate: int, number of samples per second (not used for now)
    window_size: int, number of samples per window
    Returns:
    Segmented and preprocessed data and corresponding labels
    """
    # segment data into windows
    # window = small segment of data
    segments = []
    segment_labels = []

    for i in range(0, len(data) - window_size, window_size // 2):
        segment = data[i : i + window_size]
        if len(segment) == window_size:
            segments.append(segment)
            # common label in this window
            segment_label = np.mean(labels[i : i + window_size])
            segment_labels.append(1 if segment_label >= 0.5 else 0)

    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    # normalize the data
    # helps the model learn better
    scaler = StandardScaler()
    segments_scaled = np.array(
        [scaler.fit_transform(segment.reshape(-1, 1)).flatten() for segment in segments]
    )

    print(f"Preprocessed data shape: {segments_scaled.shape}")
    print(f"Preprocessed labels shape: {segment_labels.shape}")

    return segments_scaled, segment_labels


def create_cnn_model(input_shape):
    """
    Create a 1D CNN model for signal classification
    Parameters:
    input_shape: tuple, shape of input data
    Returns:
    Compiled Keras model
    """
    model = Sequential(
        [
            Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),  # binary classification
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("Model created")
    model.summary()
    return model


def train_signal_model(X_data, y_labels, window_size=1000, sampling_rate=250):
    """
    Train the model on signal data
    Parameters:
    X_data: numpy array of signal data
    y_labels: numpy array of labels
    window_size: int, number of samples per window
    sampling_rate: int, number of samples per second
    Returns:
    Trained model and training history
    """
    # preprocess the data
    X_processed, y_processed = preprocess_signal_data(
        X_data, y_labels, sampling_rate, window_size
    )

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=0.2,
        random_state=42,
        # random_state is a seed for the random number generator
        # 20% of the data will be used for testing
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Create and train the model
    model = create_cnn_model(input_shape=(window_size, 1))

    history = model.fit(
        X_train.reshape(-1, window_size, 1),
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        X_test.reshape(-1, window_size, 1), y_test, verbose=0
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    return model, history


def predict_signal(model, new_data, window_size=1000, sampling_rate=250):
    """
    Make predictions on new signal data
    Parameters:
    model: trained Keras model
    new_data: numpy array of new signal data
    window_size: int, number of samples per window
    sampling_rate: int, number of samples per second
    Returns:
    Predictions for the input data
    """
    # Create dummy labels for preprocessing (will not be used for prediction)
    dummy_labels = np.zeros(len(new_data))

    # Preprocess new data
    processed_data, _ = preprocess_signal_data(
        new_data, dummy_labels, sampling_rate, window_size
    )

    # Make predictions
    predictions = model.predict(processed_data.reshape(-1, window_size, 1))
    return predictions


if __name__ == "__main__":
    print("Starting the program...")

    # create dummy data
    # generate 10000 samples of random signal data
    signal_length = 10000
    X_dummy = np.random.randn(signal_length)

    # generate random binary labels (0 or 1)
    y_dummy = np.random.randint(0, 2, size=signal_length)

    print(f"Created dummy data with shape: {X_dummy.shape}")
    print(f"Labels shape: {y_dummy.shape}")

    # train the model
    print("Starting model training...")
    model, history = train_signal_model(X_dummy, y_dummy)

    # make predictions on new data
    print("\nTesting predictions...")
    new_data = np.random.randn(2000)  # create some new test data
    predictions = predict_signal(model, new_data)
    print(f"Predictions shape: {predictions.shape}")

    # Print training history
    print("\nTraining History:")
    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])
