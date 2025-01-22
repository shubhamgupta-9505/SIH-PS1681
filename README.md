# SIH-PS1681

**Private Repository**

---
About

The folder final_codes contains the code for machine learning model training.

The folder SIH contains the code for the website. To run the application, simply execute app.py as mentioned below.

The folder SIH also contains the weights of the trained model.

## Requirements

### Prerequisites:
- **Python 3**
- **pip3** (Python package manager)

### Setting Up the Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   ```

2. Activate the virtual environment:
   - **For Mac/Linux**:
     ```bash
     source myenv/bin/activate
     ```
   - **For Windows**:
     ```bash
     myenv\Scripts\activate
     ```

### Installing Dependencies

Once the virtual environment is activated, run the following commands to install the required Python packages:

```bash
pip install pycryptodome sympy
pip install tensorflow
pip install xgboost
pip install scikit-learn
pip install pandas
pip install flask
pip install joblib
pip install h5py
pip install numpy
```

---

## Running the Application

With your virtual environment activated, type the following command in the same terminal to run the application 
```bash
python SIH/app.py
```

This website is configured to run locally on **port 5000**.

To start the application, ensure the virtual environment is activated and then run your Python script. Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

or

```
http://localhost:5000
```


