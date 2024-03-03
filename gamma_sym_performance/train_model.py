# %% Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import root_scalar

# custom modules
from models_binary import unet_model, lstm_model, cnn_model

# %% Load data
(X_train, Y_train, X_test, Y_test) = np.load('database_gamma.npz').values()


# %% Normalize data
def normalize_data(X):
    X = X/3000 - 1
    return X


X_train = normalize_data(X_train)
# %%
learning_rate = 0.001
sample_length = X_train.shape[1]
print('frame_length:', sample_length)

tf.keras.backend.clear_session()
# model = unet_model(depth=4, frame_length=sample_length)
model = lstm_model(lstm_depth=2, filters=16, sample_length=sample_length)
# model = cnn_model(inception_depth=2, sample_length=sample_length)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

callback_stp = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                patience=4,
                                                verbose=2,
                                                restore_best_weights=True)

callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=2, min_lr=5e-5, verbose=1)

#%% Train model
history = model.fit(X_train, Y_train, batch_size=256, epochs=30, validation_split=0.1,
                    callbacks=[callback_stp, callback_reduce_lr], verbose=1)

# Save model
model.save('model.keras')

# %% Plot training history.
pdf = PdfPages("model_loss_perf.pdf")
plt.figure(figsize=(5, 4))
plt.subplot(2, 1, 1)
plt.suptitle('Parameters: {}'.format(model.count_params()))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlim(left=1)
# plt.ylim(top=0.004, bottom=0)
plt.legend()
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
plt.title('Loss')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('best val_acc {:.4f}'.format(history.history['val_accuracy'][-1]))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.grid()
plt.xlim(left=1)
plt.ylim(bottom=0.9)

#%%
