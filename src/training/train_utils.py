import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def ctc_loss_function(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")

    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int32")

    label_length = tf.math.count_nonzero(y_true != -1, axis=1, keepdims=True, dtype="int32")

    loss = tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )
    return loss

def train_model(model, train_dataset, val_dataset, epochs, model_path):
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=ctc_loss_function)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history