import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Load embeddings
X = np.load("../dataSet/X.npy")
y = np.load("../dataSet/y.npy")

print("X:", X.shape)
print("y:", y.shape)

#Split data
X_train, X_test, y_train, y_test = train_test_split( # Split so they correlate, X[0]= siren therefore y[0]= 1
    X,  y,
    test_size=0.2, #Saves some to test model on
    random_state=42, #Makes the split reproducible (run it again & same train/test split.)
    stratify=y #Keeps the same siren/non-siren ratio in both sets. E.g. 60% 40% in original, 60% 40% in both train and test
)

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
print("Recall   :", recall)


#Save
model.save("../models/siren_classifier.keras")
