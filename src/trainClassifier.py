import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

#Load embeddings
X = np.load("../dataSet/X.npy")
y = np.load("../dataSet/y.npy")
groups = np.load("../data/groups.npy")

print("X:", X.shape)
print("y:", y.shape)

#Split data by file
splitter = GroupShuffleSplit(n_splits=1, #
    test_size=0.2, #Saves some to test model on
    random_state=42, #Makes the split reproducible (run it again & same train/test split.)
)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

#Define model, create neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)), #Each input has 1024 features (YamNet embeddings)
    tf.keras.layers.Dense(128, activation="relu"), #Learns patterns associated with sirens
    tf.keras.layers.Dropout(0.3), #Stops model memorises training data
    tf.keras.layers.Dense(32, activation="relu"), #Compact data
    tf.keras.layers.Dense(1, activation="sigmoid") #Produces either 0 or 1
])

#Compile
model.compile(
    optimizer="adam", #Adjusts during learning.
    #E.g. weights= [0.2, 0.8, 0.1] but ambulance(0.8) is wrongly classified as non-siren. Weights are then adjusted- [0.3, 1.0, 0.05]
    loss="binary_crossentropy", #Measures prediction error for yes/no classification.
    metrics=["accuracy",  #Percentage of predictions that are correct
             tf.keras.metrics.Precision(), #How often is the model correct when saying siren
             tf.keras.metrics.Recall()] #How many actual sirens it finds.
)

#Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2, #Uses 20% of training data to check progress during training.
    epochs=15, #Reads training data 15 times
    batch_size=16 #16 samples at a time
)

#Evaluate on unseen data
loss, accuracy, precision, recall = model.evaluate( #Tests the model on the unseen data
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("Precision:", precision)
print("Recall:", recall)

#Save
model.save("../models/siren_classifier.keras")

# Convert to TFLite so it can run on the Raspberry Pi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("../models/siren_classifier.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved siren_classifier.tflite")
