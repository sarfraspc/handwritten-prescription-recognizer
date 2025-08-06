import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def ctc_loss_function(y_true, y_pred):
    """
    Computes the CTC loss.

    Args:
        y_true: True labels (padded).
        y_pred: Predicted labels from the model.

    Returns:
        The CTC loss.
    """
    # Get the length of the predicted labels
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
    input_length = tf.tile([input_length], [tf.shape(y_pred)[0]])

    # Get the length of the true labels
    label_length = tf.math.count_nonzero(y_true, axis=1, dtype="int32")

    # Compute the CTC loss
    loss = tf.keras.backend.ctc_batch_cost(
        tf.cast(y_true, dtype="int32"), 
        y_pred, 
        tf.reshape(input_length, (-1, 1)), 
        tf.reshape(label_length, (-1, 1))
    )
    return loss

def train_model(model, train_dataset, val_dataset, epochs, model_path):
    """
    Trains the CRNN model.

    Args:
        model: The Keras model to train.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        epochs (int): The number of epochs to train for.
        model_path (str): The path to save the best model.

    Returns:
        The trained model and the training history.
    """
    # Compile the model with the CTC loss and Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=ctc_loss_function)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history