import tensorflow as tf
import numpy as np

class RLNPolicy:
    """Reactive Learning Network policy using TensorFlow Lite for inference."""

    def __init__(self, observ_dim, hidden_dim, action_dim, lr, model_path):
        self.model_path = model_path
        # Training model built with TensorFlow
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(observ_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(action_dim)
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Interpreter for inference
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _update_interpreter(self):
        """Updates the TFLite interpreter with weights from the training model."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def train(self, observs, actions):
        """Trains the RLN model using TensorFlow."""
        observ_tensor = tf.convert_to_tensor(observs, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred_action = self.model(observ_tensor, training=True)
            loss = self.loss_fn(action_tensor, pred_action)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self._update_interpreter()
        return float(loss.numpy())

    def get_action(self, observ):
        """Runs inference using the TFLite interpreter."""
        input_data = np.asarray(observ, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0]

