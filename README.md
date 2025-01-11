Fraud Detection System

This project implements a Fraud Detection System using a pre-trained LSTM model for detecting anomalies in financial transactions. The backend is built with FastAPI, and the UI is served using Jinja2 templates.

Features

Pre-trained LSTM model for fraud detection, built on Kaggle.

REST API for fraud prediction.

Interactive charts for fraud analysis.

Simple web-based UI.

Dataset

Name: Financial Anomaly Data

Platform: Kaggle

The dataset contains financial transaction data used to train the model for fraud detection.

Project Structure

.
├── main.py                  # FastAPI application
├── model                   # Directory for ML models
│   └── Fraud_detection_model_.h5  # Pre-trained LSTM model
├── templates               # HTML templates
│   ├── index.html          # Main Dashboard
│   ├── chart.html          # Charts and visualization
│   └── about.html          # About page
├── financial_anomaly_data.csv  # Dataset used for training and analysis
├── README.md               # Documentation (this file)
└── requirements.txt        # Python dependencies

Requirements

Prerequisites

Python 3.8+

Install Dependencies

pip install -r requirements.txt

How to Run Locally

Step 1: Verify Files

Ensure the following are available in the project directory:

main.py

model/Fraud_detection_model_.h5

financial_anomaly_data.csv

templates/ directory with HTML files.

Step 2: Install Dependencies

Run the following command to install required libraries:

pip install -r requirements.txt

Step 3: Start the Application

Start the FastAPI server:

uvicorn main:app --reload

Step 4: Access the Application

Open your browser and navigate to: http://127.0.0.1:8000

API documentation is available at: http://127.0.0.1:8000/docs

Step 5: Explore the UI

Dashboard: Root URL to see transaction analysis.

Charts Page: Navigate to /chart for visualizations.

About Page: Navigate to /about for project details.

API Endpoints

/ (GET)

Displays the main dashboard.

/predict/ (POST)

Predicts fraud for a transaction.

Request Example:

curl -X POST "http://127.0.0.1:8000/predict/" \
-H "Content-Type: application/json" \
-d '{"TransactionID": "<Transaction ID>"}'

Response Example:

{
  "TransactionID": "<Transaction ID>",
  "Fraud": true/false
}

/chart (GET)

Displays fraud analysis charts.

/about (GET)

Displays project information.

Dataset and Model

Dataset

Name: Financial Anomaly Data

Platform: Kaggle

The dataset includes transaction details, which are preprocessed for LSTM training.

Model Training

Preprocessing:

Fill missing values.

Normalize the Amount column.

Create sequences of 15 data points for LSTM input.

Training:

LSTM model trained to learn reconstruction errors.

Model exported as Fraud_detection_model_.h5.

Notes

Fraud Detection Threshold:

Fraud is detected based on reconstruction error percentile (95th percentile).

Error Handling:

Application handles invalid inputs, missing files, and empty datasets.

Scalability:

For large datasets, consider batch processing or streaming.

License

This project is licensed under the MIT License.

Acknowledgements

Dataset and Model Training: Kaggle

Framework: FastAPI

