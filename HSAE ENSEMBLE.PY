import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import psutil

def build_hsae(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input_layer)
    encoded = LeakyReLU(0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(512)(encoded)
    encoded = LeakyReLU(0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(256, name="latent_features")(encoded)
    encoded = LeakyReLU(0.1)(encoded)
    decoded = Dense(512)(encoded)
    decoded = LeakyReLU(0.1)(decoded)
    decoded = Dense(1024)(decoded)
    decoded = LeakyReLU(0.1)(decoded)
    output_layer = Dense(input_dim)(decoded)
    anomaly_score = Dense(1, activation='sigmoid', name="anomaly_score")(encoded)
    model = Model(inputs=input_layer, outputs=[output_layer, anomaly_score])
    def hybrid_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true[0] - y_pred[0]))
        classification_loss = tf.keras.losses.BinaryCrossentropy()(y_true[1], y_pred[1])
        return reconstruction_loss + 0.03 * classification_loss
    model.compile(optimizer=Adam(0.00005), loss=hybrid_loss)
    return model

hsae = build_hsae(X_train_scaled.shape[1])
y_train = [X_train_scaled, y_train_labels.reshape(-1, 1)]
hsae.fit(X_train_scaled, y_train, epochs=150, batch_size=128, validation_split=0.2, verbose=1, callbacks=[early_stop])

encoder_model = Model(inputs=hsae.input, outputs=hsae.get_layer("latent_features").output)
X_train_latent = encoder_model.predict(X_train_scaled)
X_val_latent = encoder_model.predict(X_val_scaled)
X_test_latent = encoder_model.predict(X_test_scaled)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_latent)
X_val_pca = pca.transform(X_val_latent)
X_test_pca = pca.transform(X_test_latent)

ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.045)
ocsvm.fit(X_train_pca)

ocsvm_scores_val = -ocsvm.decision_function(X_val_pca)
ocsvm_scores_val_norm = (ocsvm_scores_val - np.min(ocsvm_scores_val)) / (np.max(ocsvm_scores_val) - np.min(ocsvm_scores_val))
ocsvm_scores_test = -ocsvm.decision_function(X_test_pca)
ocsvm_scores_test_norm = (ocsvm_scores_test - np.min(ocsvm_scores_test)) / (np.max(ocsvm_scores_test) - np.min(ocsvm_scores_test))

X_val_reconstructed = hsae.predict(X_val_scaled)[0]
reconstruction_errors_val = np.mean(np.square(X_val_scaled - X_val_reconstructed), axis=1)
reconstruction_errors_val_norm = reconstruction_errors_val / np.max(reconstruction_errors_val)

X_test_reconstructed = hsae.predict(X_test_scaled)[0]
reconstruction_errors_test = np.mean(np.square(X_test_scaled - X_test_reconstructed), axis=1)
reconstruction_errors_test_norm = reconstruction_errors_test / np.max(reconstruction_errors_test)

combined_scores_val = 0.5 * reconstruction_errors_val_norm + 0.5 * ocsvm_scores_val_norm
combined_scores_test = 0.5 * reconstruction_errors_test_norm + 0.5 * ocsvm_scores_test_norm

fpr_val, tpr_val, thresholds_roc_val = roc_curve(y_val_labels, combined_scores_val)
fnr_val = 1 - tpr_val
eer_index = np.argmin(np.abs(fpr_val - fnr_val))
eer_threshold = thresholds_roc_val[eer_index]
y_pred_binary = (combined_scores_test >= eer_threshold).astype(int)
