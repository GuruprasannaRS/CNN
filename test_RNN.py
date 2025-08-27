# Imported required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Loaded your saved model
    model = load_model('model/21051082_RNN_model.h5')

    # 2. Loaded your testing data
    test_data = pd.read_csv('data/test_data_RNN.csv')
    
    # Separated features and target
    features = test_data[['Open_1', 'High_1', 'Low_1', 'Volume_1',
                          'Open_2', 'High_2', 'Low_2', 'Volume_2',
                          'Open_3', 'High_3', 'Low_3', 'Volume_3']].values
    target = test_data['Next_Open'].values
    
    # Normalized the features using the same scaler from training
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Reshaped features for input to LSTM (samples, timesteps, features)
    features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 3, 4)) 

    # 3. Ran prediction on the test data and output required plot and loss
    scores = model.evaluate(features_reshaped, target)
    pred = model.predict(features_reshaped)
    print('Test Mean squared error: ', scores[0])
    print('Test Mean absolute error: ', scores[1])
	
    # Defined the marker styles for true and predicted values
    true_marker = 'o'
    predicted_marker = 's'

    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotted true and predicted values with markers and labels
    ax.plot(target, label='True Stock Price', marker=true_marker, color='blue')
    ax.plot(pred, label='Predicted Stock Price', marker=predicted_marker, color='red')

    # Set the x-axis, y-axis labels, titles and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Opening Price')
    ax.set_title('True vs Predicted Stock Price')
    ax.legend()

    # Displayed the plot
    plt.show()
