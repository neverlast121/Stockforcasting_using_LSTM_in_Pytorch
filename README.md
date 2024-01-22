# Time Series Analysis with LSTM using PyTorch

This Python script utilizes PyTorch to implement a Long Short-Term Memory (LSTM) neural network for time series analysis on stock prices. The code involves data preprocessing, model training, and visualization of the predicted and actual stock prices.

## Dependencies

- `pandas`: Library for data manipulation and analysis.
- `numpy`: Library for numerical operations.
- `matplotlib.pyplot`: Library for creating visualizations.
- `torch`: PyTorch library for deep learning.

## Code Overview

1. **Data Loading:**
   - Read historical stock price data from a CSV file (`AMZN.csv`) using pandas.
   - Display the initial data.

2. **Data Preprocessing:**
   - Extract relevant columns (`Date` and `Close`) from the dataset.
   - Convert the `Date` column to datetime format.
   - Plot the stock prices over time.

3. **Feature Engineering:**
   - Create a function `prepare_dataframe_lstm` to transform the time series data into a format suitable for LSTM training.
   - Shift the 'Close' prices for the last 7 days to create features.
   - Scale the data using MinMaxScaler.

4. **Data Splitting:**
   - Split the dataset into training and testing sets.

5. **Data Reshaping:**
   - Reshape the data into the required input shape for LSTM.

6. **Data Conversion to PyTorch Tensors:**
   - Convert the NumPy arrays to PyTorch tensors.

7. **Dataset and DataLoader Setup:**
   - Define a custom dataset class (`TimeseriesData`) for handling the training and testing data.
   - Create DataLoader objects for efficient batch processing.

8. **LSTM Model Definition:**
   - Define an LSTM model class (`LSTM`) using PyTorch's `nn.Module`.

9. **Model Training:**
   - Train the LSTM model for 10 epochs using Mean Squared Error (MSE) loss.
   - Print training loss for every 100 batches.

10. **Model Validation:**
    - Validate the trained model on the test dataset and print the validation loss.

11. **Prediction and Visualization:**
    - Generate predictions on the training set and visualize actual vs predicted stock prices.
    - Generate predictions on the testing set and visualize actual vs predicted stock prices.

12. **Visualization of Training Predictions:**
    - Plot the predicted and actual stock prices on the training set.

13. **Visualization of Testing Predictions:**
    - Plot the predicted and actual stock prices on the testing set.

## Note
- The script assumes the existence of a CSV file named `AMZN.csv` containing stock price data.
- The LSTM model is defined with an input size of 1, hidden size of 4, and a single stacked layer. These hyperparameters can be adjusted based on the specific use case.

