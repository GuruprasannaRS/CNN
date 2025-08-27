# Import required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Load your training data
    data = pd.read_csv('data/q2_dataset.csv')

    # List of features used in the model
    features_list = [' Open', ' High', ' Low', ' Volume']
    features = data[features_list].values
    target = data[' Open'].values

    # Initialized lists to store the features and target
    X = []
    y = []

    # Created the dataset by using the latest 3 days as features and the next day's opening price as the target
    for i in range(len(features) - 3):
        # Selected the past 3 days features and the next day's opening price
        x_i = features[i:i+3]
        x_i = x_i.flatten()  # Flattened the 2D array to a 1D array
        y_i = target[i+3]

        # Added the features and target to the respective lists
        X.append(x_i)
        y.append(y_i)

    # Converted the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Randomized the data
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Splited the data into training and testing sets
    train_ratio = 0.7
    train_size = int(train_ratio * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]

    X_test = X_shuffled[train_size:]
    y_test = y_shuffled[train_size:]

    # Created dataframes for the training and testing sets
    train_data = pd.DataFrame(X_train, columns=['Open_1', 'High_1', 'Low_1', 'Volume_1',
                                                'Open_2', 'High_2', 'Low_2', 'Volume_2',
                                                'Open_3', 'High_3', 'Low_3', 'Volume_3'
                                                ])
    train_data['Next_Open'] = y_train

    test_data = pd.DataFrame(X_test, columns=['Open_1', 'High_1', 'Low_1', 'Volume_1',
                                              'Open_2', 'High_2', 'Low_2', 'Volume_2',
                                              'Open_3', 'High_3', 'Low_3', 'Volume_3'
                                              ])
    test_data['Next_Open'] = y_test

    # Saved the training and testing data to CSV files
    train_data.to_csv('data/train_data_RNN.csv', index=False)
    test_data.to_csv('data/test_data_RNN.csv', index=False)
    print("Saved Successfully")

    # 2. Train your network
    train_data = pd.read_csv('data/train_data_RNN.csv')
    
    # Separated features and target
    features = train_data[['Open_1', 'High_1', 'Low_1', 'Volume_1',
                           'Open_2', 'High_2', 'Low_2', 'Volume_2',
                           'Open_3', 'High_3', 'Low_3', 'Volume_3'
                           ]].values
    target = train_data['Next_Open'].values

    print('Shape of the features:', features.shape)
    print('Shape of the target:', target.shape)

    # Normalized the features using Min-Max scaling to a range of [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Reshaped features for input to LSTM (samples, timesteps, features)
    features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 3, 4))
	

    # Created the LSTM model
    model = Sequential()
    
	# Added the first LSTM layer with 256 units, return sequences to pass output to the next layer, and input shape of (3, 4) corresponding to 3 time steps and 4 features for each time step
	# Added a dropout layer with a rate of 0.2 to prevent overfitting by randomly setting 20%
	# Added the second LSTM layer with 256 units, return sequences to pass output to the next layer
	# Added a dropout layer with a rate of 0.2 to prevent overfitting by randomly setting 20%
	# Added the third LSTM layer with 256 units. This layer processes the final sequence and returns the last output
	# Added a dropout layer with a rate of 0.2 to prevent overfitting by randomly setting 20%
	# Added the output layer with a single unit (Dense(1))

    model.add(LSTM(units=256, return_sequences=True, input_shape=(3, 4)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()

    # Compiled the model with Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metric
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Train the model
    history = model.fit(features_reshaped, target, epochs=500, batch_size=64, validation_split=0.2)

    # Get the training and validation MSE (Mean Squared Error) from the history object
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plotting the Train and Validation loss values over epochs
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Get the training and validation MAE (Mean Absolute Error) from the history object
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    # Plotting the Train and Validation MAE values over epochs
    plt.plot(epochs, train_mae, 'b', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    # Printing the final MAE and MSE values
    print("Training MAE:", train_mae[-1])
    print("Validation MAE:", val_mae[-1])
    print("Training MSE:", train_loss[-1])
    print("Validation MSE:", val_loss[-1])

    # 3. Save your model
    model.save("model/21051082_RNN_model.h5")
    print('Model Saved Successfully')
