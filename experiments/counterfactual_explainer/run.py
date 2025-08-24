import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
import os

# Suppress TensorFlow logging for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# --- 1. Data Generation and Model Setup ---


def generate_synthetic_scene_data(n_samples=10000):
    """Generates a synthetic dataset representing two objects in a 2D scene."""
    # Object 1 positions are random
    obj1_pos = np.random.rand(n_samples, 2) * 100

    # Object 2's position is correlated with Object 1's, introducing a dependency
    obj2_pos = obj1_pos + np.random.randn(n_samples, 2) * 15
    obj2_pos = np.clip(obj2_pos, 0, 100)

    # Features: [x1, y1, x2, y2]
    X = np.hstack([obj1_pos, obj2_pos]).astype(np.float32)

    # Target label: Is object 1 to the left of object 2?
    y = (X[:, 0] < X[:, 2]).astype(np.int32)
    return X, y


def create_classifier(input_shape):
    """Creates and compiles a simple MLP classifier."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# --- 2. RealAC Core Components (Adapted for Synthetic Data) ---


def estimate_mi(x, y, bins=30):
    """Estimates mutual information between two continuous variables."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    p_xy = hist_2d / (np.sum(hist_2d) + 1e-6) + 1e-6
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = np.outer(p_x, p_y) + 1e-6
    valid = (p_xy > 1e-6) & (p_x_p_y > 1e-6)
    mi = np.sum(p_xy[valid] * np.log(p_xy[valid] / p_x_p_y[valid]))
    return np.clip(mi, 0, 5)


class MILossLayer(keras.layers.Layer):
    def __init__(self, mi_original_tf, lambda_mi=1.0, bins=30, **kwargs):
        super(MILossLayer, self).__init__(**kwargs)
        self.mi_original_tf = mi_original_tf
        self.lambda_mi = lambda_mi
        self.bins = bins

    def call(self, inputs):
        x_recon = inputs
        mi_loss = 0.0
        for (i, j), mi_orig in self.mi_original_tf.items():
            x_i, x_j = x_recon[:, i], x_recon[:, j]

            # TensorFlow-based histogram2d for MI calculation in-graph
            hist_2d = tf.histogram2d(
                x_i, x_j, bins=self.bins, range=[[0.0, 100.0], [0.0, 100.0]]
            )
            p_xy = hist_2d / (tf.reduce_sum(hist_2d) + 1e-6) + 1e-6
            p_x = tf.reduce_sum(p_xy, axis=1)
            p_y = tf.reduce_sum(p_xy, axis=0)
            p_x_p_y = tf.expand_dims(p_x, 1) * tf.expand_dims(p_y, 0) + 1e-6

            valid = tf.cast((p_xy > 1e-6) & (p_x_p_y > 1e-6), tf.float32)
            mi = tf.reduce_sum(p_xy * tf.math.log(p_xy / p_x_p_y) * valid)
            mi_loss += tf.abs(mi - mi_orig)

        self.add_loss(self.lambda_mi * mi_loss / (len(self.mi_original_tf) + 1e-6))
        return x_recon


class VAE(keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        clf,
        lambda_kl,
        lambda_clf,
        lambda_mse,
        lambda_fixed,
        **kwargs,
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.clf = clf
        self.lambda_kl = lambda_kl
        self.lambda_clf = lambda_clf
        self.lambda_mse = lambda_mse
        self.lambda_fixed = lambda_fixed
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.clf_loss_tracker = keras.metrics.Mean(name="clf_loss")

    def train_step(self, data):
        (x_orig, t_target, mask), _ = data
        with tf.GradientTape() as tape:
            mu, log_var = self.encoder(x_orig)
            z = mu + tf.exp(0.5 * log_var) * tf.random.normal(shape=tf.shape(mu))
            x_recon = self.decoder([z, t_target])

            # Counterfactual is a mix of original (fixed) and reconstructed (mutable)
            cf = mask * x_orig + (1 - mask) * x_recon

            # Loss components
            recon_loss_mutable = tf.reduce_mean(
                tf.reduce_sum((1 - mask) * tf.square(x_orig - x_recon), axis=1)
            )
            recon_loss_fixed = tf.reduce_mean(
                tf.reduce_sum(mask * tf.square(x_orig - x_recon), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var)
            )
            clf_loss = tf.keras.losses.sparse_categorical_crossentropy(
                t_target, self.clf(cf)
            )
            mi_loss = sum(self.decoder.losses)

            total_loss = (
                self.lambda_mse * recon_loss_mutable
                + self.lambda_fixed * recon_loss_fixed
                + self.lambda_kl * kl_loss
                + self.lambda_clf * clf_loss
                + mi_loss
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss_mutable)
        self.kl_loss_tracker.update_state(kl_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        return {m.name: m.result() for m in self.metrics}


# --- 3. Main Execution ---
if __name__ == "__main__":
    # Hyperparameters
    N_FEATURES = 4
    LATENT_DIM = 8
    LAMBDA_KL = 0.1
    LAMBDA_CLF = 10.0
    LAMBDA_MSE = 5.0
    LAMBDA_FIXED = 20.0
    LAMBDA_MI = 2.0

    print("1. Generating synthetic data and training classifier...")
    X, y = generate_synthetic_scene_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = create_classifier(N_FEATURES)
    classifier.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    accuracy = accuracy_score(
        y_test, np.argmax(classifier.predict(X_test, verbose=0), axis=1)
    )
    print(f"   -> Classifier Test Accuracy: {accuracy:.4f}")
    classifier.trainable = False

    print("2. Building and training the Counterfactual VAE model...")
    # Pre-compute original MI to use as a target for the realism loss
    mi_original = {}
    for i, j in combinations(range(N_FEATURES), 2):
        mi_original[(i, j)] = estimate_mi(X_train[:, i], X_train[:, j])
    mi_original_tf = {
        k: tf.constant(v, dtype=tf.float32) for k, v in mi_original.items()
    }

    # VAE Encoder
    encoder_inputs = keras.Input(shape=(N_FEATURES,))
    x = keras.layers.Dense(64, activation="relu")(encoder_inputs)
    x = keras.layers.Dense(32, activation="relu")(x)
    z_mean = keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    # VAE Decoder
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    target_class_inputs = keras.Input(shape=(1,))
    merged_inputs = keras.layers.Concatenate()([latent_inputs, target_class_inputs])
    x = keras.layers.Dense(32, activation="relu")(merged_inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    decoder_outputs = keras.layers.Dense(N_FEATURES)(x)
    decoder_outputs = MILossLayer(mi_original_tf, LAMBDA_MI)(
        decoder_outputs
    )  # Add realism loss layer
    decoder = keras.Model(
        [latent_inputs, target_class_inputs], decoder_outputs, name="decoder"
    )

    # VAE Model
    vae = VAE(
        encoder, decoder, classifier, LAMBDA_KL, LAMBDA_CLF, LAMBDA_MSE, LAMBDA_FIXED
    )
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    # Prepare data for VAE training
    target_train = 1 - y_train
    mask_train = np.zeros_like(X_train)
    # For this demo, we make obj1's features mutable and obj2's fixed for 50% of the data
    mutable_indices = np.random.choice([True, False], size=len(X_train))
    mask_train[mutable_indices, 2:] = 1.0  # Fix obj2
    mask_train[~mutable_indices, :2] = 1.0  # Fix obj1

    vae.fit(
        [X_train, target_train.reshape(-1, 1), mask_train],
        X_train,
        epochs=20,
        batch_size=128,
        verbose=1,
    )

    print("\n3. Generating and evaluating a counterfactual explanation...")
    # Select a sample to explain
    sample_idx = 0
    x_factual = X_test[sample_idx : sample_idx + 1]
    y_factual_pred_prob = classifier.predict(x_factual, verbose=0)
    y_factual_pred = np.argmax(y_factual_pred_prob, axis=1)[0]

    # Define actionability: we want to find a CF by only moving object 2 (features 2, 3)
    # Mask = 1 for features we want to keep fixed.
    actionability_mask = np.array([[1.0, 1.0, 0.0, 0.0]])
    target_class = np.array([[1 - y_factual_pred]])

    # Generate counterfactual
    mu, log_var = vae.encoder.predict(x_factual, verbose=0)
    z = mu  # Use mean for deterministic generation
    x_recon = vae.decoder.predict([z, target_class], verbose=0)
    x_counterfactual = (
        actionability_mask * x_factual + (1 - actionability_mask) * x_recon
    )
    y_cf_pred_prob = classifier.predict(x_counterfactual, verbose=0)
    y_cf_pred = np.argmax(y_cf_pred_prob, axis=1)[0]

    # Print results
    pred_map = {0: "'Obj1 is NOT left of Obj2'", 1: "'Obj1 is LEFT of Obj2'"}
    print(f"\n--- COUNTERFACTUAL EXPLANATION ---")
    print(f"Factual Input (Obj1, Obj2 positions): {np.round(x_factual[0], 2)}")
    print(
        f"  -> Model Prediction: {pred_map[y_factual_pred]} (Confidence: {y_factual_pred_prob[0, y_factual_pred]:.2f})"
    )
    print(f"\nActionability Constraint: Keep Object 1's position fixed.")
    print(f"Desired Outcome: Flip prediction to {pred_map[1 - y_factual_pred]}.")
    print(f"\nCounterfactual (New positions): {np.round(x_counterfactual[0], 2)}")
    print(
        f"  -> Model Prediction: {pred_map[y_cf_pred]} (Confidence: {y_cf_pred_prob[0, y_cf_pred]:.2f})"
    )

    if y_cf_pred != y_factual_pred:
        print(
            "\nSUCCESS: The counterfactual successfully flipped the model's prediction."
        )
    else:
        print("\nFAILURE: The counterfactual did not flip the model's prediction.")

    fixed_features_changed = (
        np.sum(np.abs(x_factual - x_counterfactual) * actionability_mask) > 1e-4
    )
    if not fixed_features_changed:
        print("SUCCESS: The actionability constraint (fixed features) was respected.")
    else:
        print("FAILURE: The fixed features were altered during CF generation.")
