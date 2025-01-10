from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
import os
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your pre-trained LSTM model
model_path = 'model/Fraud_detection_model_.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Load the dataset
dataset_path = 'financial_anomaly_data.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
df = pd.read_csv(dataset_path)

# Validate the dataset
if df.empty:
    raise ValueError("The dataset is empty. Please provide a valid dataset.")

# Handle missing values (fill NaNs with 0 or mean value for numeric columns)
df.fillna(0, inplace=True)  # Alternatively: df.fillna(df.mean(), inplace=True)

print(f"Shape of DataFrame after handling NaNs: {df.shape}")

# Ensure required columns exist
required_columns = ['TransactionID', 'Amount']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Limit to first 10000 rows for processing
df = df.head(10000)

# Normalize the Amount column
scaler = MinMaxScaler()
if not df['Amount'].empty:
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
else:
    raise ValueError("The 'Amount' column is empty. Cannot scale data.")

# Create sequences for LSTM
sequence_length = 15

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Prepare data sequences for LSTM
all_data_sequences = create_sequences(df['Amount_scaled'].values, sequence_length)
if all_data_sequences.size == 0:
    raise ValueError("Not enough data to create sequences. Ensure the dataset has sufficient rows.")

all_data_sequences_reshaped = all_data_sequences.reshape((all_data_sequences.shape[0], sequence_length, 1))

# Predict using the model
all_data_pred = model.predict(all_data_sequences_reshaped)

# Flatten the third dimension of `all_data_sequences_reshaped`
all_data_sequences_flat = all_data_sequences_reshaped.squeeze(axis=-1)

# Calculate Mean Absolute Error (MAE) loss for each sequence
all_data_mae_loss = np.mean(np.abs(all_data_pred - all_data_sequences_flat), axis=1)

# Determine threshold for fraud detection
threshold = np.percentile(all_data_mae_loss, 95)

# Set fraud or non-fraud status based on MAE loss and threshold
fraud_status = ["FRAUD" if error > threshold else "NON-FRAUD" for error in all_data_mae_loss]

# Create a DataFrame for analysis
analysis_df = pd.DataFrame({
    'TransactionID': df.iloc[sequence_length:]['TransactionID'].values,
    'ReconstructionError': all_data_mae_loss,
    'Status': fraud_status
})

# Extract Transaction IDs for Fraud and Non-Fraud
fraudulent_transactions = analysis_df[analysis_df['Status'] == "FRAUD"]['TransactionID'].tolist()
non_fraudulent_transactions = analysis_df[analysis_df['Status'] == "NON-FRAUD"]['TransactionID'].tolist()

# Prepare data for the charts (bar chart and scatter plot)
fraud_count = len(fraudulent_transactions)
non_fraud_count = len(non_fraudulent_transactions)

# Scatter plot data for the transactions
scatter_data = [
    {"x": idx, "y": error, "tid": transaction_id, "type": status}
    for idx, (transaction_id, error, status) in enumerate(zip(analysis_df['TransactionID'], all_data_mae_loss, fraud_status))
]

# Prepare JSON data for the chart API
data = {
    "fraud_count": fraud_count,
    "non_fraud_count": non_fraud_count,
    "scatter_data": scatter_data
}

# Set up templates directory for Jinja2
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class Transaction(BaseModel):
    TransactionID: str


from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling features before prediction
@app.post("/predict/")
async def predict(transaction: Transaction):
    try:
        # Log the incoming transaction ID
        print(f"Received TransactionID: {transaction.TransactionID}")

        # Strip leading/trailing spaces and convert to uppercase for comparison
        transaction.TransactionID = transaction.TransactionID.strip().upper()
        df['TransactionID'] = df['TransactionID'].str.strip().str.upper()

        # Log available transaction IDs in the dataset
        print(f"Searching for TransactionID: {transaction.TransactionID}")
        
        # Check if the TransactionID exists in the dataset
        transaction_data = df[df['TransactionID'] == transaction.TransactionID]
        
        if transaction_data.empty:
            # Return clear message if TransactionID is not found
            raise HTTPException(status_code=404, detail=f"TID '{transaction.TransactionID}' Not found in the dataset")
        
        # Handle Date/Timestamp columns if needed (replace 'DateColumnName' with your actual column name)
        if 'DateColumnName' in transaction_data.columns:  # Replace 'DateColumnName' with the actual column name
            transaction_data['DateColumnName'] = pd.to_datetime(transaction_data['DateColumnName']).astype(np.int64)

        # Select only numeric columns for prediction (excluding TransactionID)
        transaction_data = transaction_data.select_dtypes(include=[np.number])

        # Ensure the data has enough features for the sequence length
        num_features = len(transaction_data.columns)
        sequence_length = 15  # Define your expected sequence length
        if num_features < sequence_length:
            # Option 1: Pad the features if not enough data (using zeros or another method)
            print(f"Not enough data points. Found {num_features}, padding with zeros to {sequence_length}.")
            padded_features = np.pad(transaction_data.values.flatten(), (0, sequence_length - num_features), mode='constant')
        elif num_features == sequence_length:
            # If exact length, no padding needed
            padded_features = transaction_data.values.flatten()
        else:
            # If there are more than 15 features, truncate to 15
            padded_features = transaction_data.values.flatten()[:sequence_length]

        # Ensure the length is exactly sequence_length (15)
        padded_features = padded_features[:sequence_length]  # Truncate if any excess

        # Log the padded features and their type
        print(f"Padded features: {padded_features}")
        print(f"Padded features data type: {padded_features.dtype}")

        # Ensure features are in the correct format for prediction
        features_array = np.array(padded_features).reshape(1, sequence_length, 1)

        # Standardize the features based on the training dataset (use the same scaler from training)
        scaler = StandardScaler()
        features_array = scaler.fit_transform(features_array.reshape(-1, 1)).reshape(1, sequence_length, 1)

        # Make prediction
        prediction = model.predict(features_array)

        # Log the prediction result
        print(f"Prediction result: {prediction}")

        # Process the prediction result (assuming the model output is a probability between 0 and 1)
        is_fraud = int(prediction[0][0] > 0.5)  # Adjust this according to your model's output
        return {"TransactionID": transaction.TransactionID, "Fraud": bool(is_fraud)}
    
    except Exception as e:
        # Log the exception
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")


@app.get("/api/get_fraud_data")
def get_fraud_data():
    return JSONResponse(content=data)

@app.get("/chart/data")
def get_chart_data():
    return JSONResponse(content=data)

@app.get("/chart", response_class=HTMLResponse)
async def chart_page(request: Request):
    return templates.TemplateResponse("chart.html", {
        "request": request,
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "scatter_data": scatter_data,
    })

from fastapi.responses import FileResponse
@app.get("/about")
async def about_page():
    return FileResponse("templates/about.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
