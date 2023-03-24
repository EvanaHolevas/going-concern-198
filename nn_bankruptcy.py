#!/usr/bin/env python
# coding: utf-8

# In[138]:


import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# Model improvement and Evaluation 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

print('tensorflow version', tf.__version__)
bank_data = pd.read_csv('data/1_american_dataset.csv')

class_names = ['Bankrupt', "Healthy"]

# data_features = pd.read_csv('data/bankrupt_features.csv')
# data_status = pd.read_csv('data/status_bankrupt.csv')


# In[139]:


bank_data['status'].value_counts()


# In[140]:


neg, pos = np.bincount(bank_data['status'])
total = neg + pos
print('Sample distributions:\n    Total: {}\n    Not Bankrupt (0): {} ({:.2f}% of total)\n    Bankrupt (1): {} ({:.2f}% of total)'.format(
    total, neg, (100 * neg) / total, pos, (100 * pos) / total))


# ## Remove entries from the majority class to balance the data.

# In[141]:


indices = bank_data[bank_data['status'] == 0].index
random_indices = np.random.choice(indices, size=neg-pos, replace=False)
bank_balanced = bank_data.drop(random_indices)

counts = bank_balanced['status'].value_counts()
print(counts)


# In[142]:


from sklearn.model_selection import train_test_split

numerical_only = bank_balanced.select_dtypes(include=['float64','int64'])
X = numerical_only.drop(['cik', 'fyear', 'status'], axis=1)
y = numerical_only['status']


# In[7]:


# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(X)
# normalizer(X)


# ## Split data into training, testing, and validation

# In[143]:


# split data into training (70%), validation (15%), and testing (15%) sets 
X_, X_test, y_, y_test = train_test_split(X, y, train_size=0.8, test_size=0.15, random_state=42, shuffle=True)
X_train, X_validate, y_train, y_validate = train_test_split(X_, y_, train_size=0.82, test_size=0.18, random_state=42, shuffle=True)


# In[144]:


print("y_train data distribution:\n", y_train.value_counts())
print("y_validate data distribution:\n", y_validate.value_counts())
print("y_test data distribution:\n", y_validate.value_counts())

input_shape = (X.shape[1],)


# ## Model 1

# In[145]:


def bankruptcy_model():
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(18, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(9, activation='relu', input_shape=input_shape),
        # tf.keras.layers.Dense(4, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(optimizer='adam', 
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'])
    return model

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss'),
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    # tf.keras.callbacks.ModelCheckpoint(filepath='./best_model.h5', monitor='val_loss', save_best_only=True)
]


# In[153]:


model = bankruptcy_model()
history = model.fit(X_train, y_train,
                    validation_data=(X_validate, y_validate),
                    epochs=200,
                    batch_size=25,
                    callbacks=callbacks,
                    # validation_steps=1,
                    # validation_batch_size=1504
                    # class_weight=class_weight
                    )


# In[148]:


def plot_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Training Set', 'Validation Set'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Training Set', 'Validation Set'], loc='upper left')

    plt.show()


# In[149]:


model.summary()


# In[154]:


plot_history(history)


# In[17]:


def plot_cm(cm):
    # Define class labels
    class_labels = ["Not Bankrupt", "Bankrupt"]

    # Create heatmap
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)

    # Add labels and title
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")

    # Show plot
    plt.show()


# In[18]:


y_predict_probs = model.predict(X_test)
print(y_predict_probs)

# Convert probabilities to binary predictions using a threshold of 0.5
y_predict = np.round(y_predict_probs)
print(y_predict)


# In[66]:


cm = confusion_matrix(y_test, y_predict)
print(cm)
plot_cm(cm)


# In[67]:


# Calculate F1 score
f1 = f1_score(y_test, y_predict)
print("F1 score: {:.2f}".format(f1))


# ## Model 2

# In[41]:


def bankruptcy_model_v2(input_shape=(18,)):
    # Define the normalization layer
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization()

    # # Adapt the normalization layer to the data
    # data = tf.ones((1, 18))
    # normalization_layer(data)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.Dense(18, activation='relu', input_shape=input_shape),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(9, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss')
    ]
    return model


# In[42]:


model2 = bankruptcy_model_v2()
history2 = model2.fit(X_train, y_train,
                    validation_data=(X_validate, y_validate),
                    epochs=100,
                    batch_size=25,
                    # callbacks=callbacks,
                    # class_weight=class_weight
                    )


# In[43]:


plot_history(history2)


# In[49]:


y_predict_probs2 = model2.predict(X_test)
y_predict2 = np.round(y_predict_probs2)


# In[52]:


# plot confusion matrix
cm_2 = confusion_matrix(y_test, y_predict2)
print(cm_2)
plot_cm(cm_2)

# Calculate F1 score
f1 = f1_score(y_test, y_predict2)
print("F1 score: {:.2f}".format(f1))


# ## Model 3

# In[176]:


def binary_classification_model(input_shape=(18,)):
    # # Define the normalization layer
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization()

    # # Adapt the normalization layer to the data
    # data = tf.ones((1, 18))
    # normalization_layer(data)
    
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss')
    ]
    
    return model


# In[173]:


y_hot_train = tf.one_hot(y_train, 2)
y_hot_val = tf.one_hot(y_validate, 2)
y_hot_test = tf.one_hot(y_test, 2)


# In[175]:


model3 = binary_classification_model()
history3 = model3.fit(X_train, y_hot_train,
                    validation_data=(X_validate, y_hot_val),
                    epochs=100,
                    batch_size=24,
                    callbacks=callbacks,
                    # class_weight=class_weight
                    )


# In[177]:


plot_history(history3)


# In[53]:


y_predict_probs3 = model2.predict(X_test)
y_predict3 = np.round(y_predict_probs3)


# In[54]:


# plot confusion matrix
cm_3 = confusion_matrix(y_test, y_predict3)
print(cm_3)
plot_cm(cm_3)

# Calculate F1 score
f1 = f1_score(y_test, y_predict3)
print("F1 score: {:.2f}".format(f1))


# In[41]:


# def binary_classification_model_4(input_shape=(18,)):
#     # create input layer
#     inputs = tf.keras.Input(shape=input_shape)
    
#     # create normalization layer
#     normalization = tf.keras.layers.experimental.preprocessing.Normalization()
#     x = normalization(inputs)
    
#     # create hidden layers
#     x = tf.keras.layers.Dense(18, activation='relu')(x)
#     x = tf.keras.layers.Dense(9, activation='relu')(x)
#     # create output layer
#     outputs = tf.keras.layers.Dense(1, activation='softmax')(x)
    
#     # create model
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     # compile model
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     return model


# ## Model 4 - Softmax

# In[160]:


def binary_classification_model_4(input_shape=(18,)):
    # # Define the normalization layer
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization()

    # # Adapt the normalization layer to the data
    # data = tf.ones((1, 18))
    # normalization_layer(data)
    
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.Dense(18, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(9, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss')
    ]
    
    return model


# In[158]:


y_hot_train = tf.one_hot(y_train, 2)
y_hot_val = tf.one_hot(y_validate, 2)
y_hot_test = tf.one_hot(y_test, 2)


# In[163]:


model4 = binary_classification_model_4()
history4 = model4.fit(X_train, y_hot_train,
                    validation_data=(X_validate, y_hot_val),
                    epochs=100,
                    batch_size=24,
                    callbacks=callbacks,
                    # class_weight=class_weight
                    )


# In[162]:


plot_history(history4)


# In[135]:


y_pred_probs4 = model.predict(X_test)
print(y_pred_probs4)

y_predict4 = np.round(y_pred_probs4)# Convert probabilities to binary predictions using a threshold of 0.5
print (y_predict4)
# print(y_predict4)


# In[136]:


# conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
num_classes = 2

# from lable to categorial
y_prediction = np.array([1,2]) 
y_categorial = tf.keras.utils.to_categorical(y_predict4, num_classes)

# from categorial to lable indexing
y_pred = y_categorial.argmax(1)

print(y_pred)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
plot_cm(conf_mat)


# In[ ]:




