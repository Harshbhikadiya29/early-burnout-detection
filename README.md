# How to Run the Burnout Detection Project

This project implements an early burnout detection system using machine learning and a Flask web application. Follow the steps below to run the code successfully.

--------------------------------------------------

STEP 1: Clone the Repository

git clone https://github.com/your-username/early-detection-of-burnout.git
cd early-detection-of-burnout

--------------------------------------------------

STEP 2: (Optional) Create and Activate Virtual Environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python -m venv venv
source venv/bin/activate

--------------------------------------------------

STEP 3: Install Required Libraries

pip install -r requirements.txt

--------------------------------------------------

STEP 4: Run the Flask Application

python app.app

--------------------------------------------------

STEP 5: Open the Application in Browser

Open a web browser and go to:
http://127.0.0.1:5000/

--------------------------------------------------

STEP 6: Use the Application

- Enter the required numerical inputs
- Enter the text input if provided
- Submit the form to receive a burnout risk prediction

--------------------------------------------------

OPTIONAL: Retrain the Model

If you want to retrain the model from scratch, run:
python train_model.py

This will preprocess the data, train the model, and save it for use in the Flask application.

--------------------------------------------------

NOTES

- Make sure Python 3 is installed on your system.
- This application is for educational purposes only and is not a medical diagnostic tool.

