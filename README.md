# Urban Traffic Flow Forecasting - Porto

This project aims to solve a Kaggle competition challenge: predicting road traffic congestion levels in the city of Porto using historical data. Following the **CRISP-DM** methodology, the project involves extensive data preprocessing, feature engineering, and the implementation of advanced ensemble machine learning models.

## Context and Objectives
Urban mobility faces significant challenges in traffic management. This project aims to predict traffic flow (categorized into 5 levels: *None, Low, Medium, High, Very High*) at a given hour to support urban planning and decision-making.

The main objectives were:
* Analyze patterns in a dataset covering over a year of traffic in Porto.
* Extract meaningful insights from environmental and temporal variables.
* Develop a predictive model with high generalization capability (avoiding overfitting).

## Some of the things done
The project followed a rigorous pipeline across several dataset versions (v1-v12):
* **Data Enrichment:** Integrated external weather data via the **Open-Meteo API** to provide precise precipitation metrics.
* **Advanced Imputation:** Handled missing values using **KNN Imputation** and **Temporal Interpolation** (Spline/Linear).
* **Feature Engineering:** * **Cyclic Encoding:** Transformed hours and weekdays into Sine/Cosine components to preserve temporal periodicity.
    * **Interaction Terms:** Created features like `Congestion_Factor` and non-linear terms like `AVERAGE_TIME_DIFF_SQR`.
    * **Temporal Features:** Extracted specific flags for holidays, rush hours, and seasonality.
* **Modeling:** * Implemented a **Stacking Ensemble** (Meta-learner) combining Random Forest, XGBoost, SVM, and MLP.
    * Hyperparameter optimization using **GridSearchCV**.
    * Model interpretability through **Permutation Importance** and **MDI**.

## Tech Stack
* **Language:** Python 3.13
* **Libraries:** requirements.txt
* **External APIs:** Open-Meteo (Weather Data)

## Prerequisites
Ensure you have the following installed:
* Python (>= 3.13)
* pip (Python package manager)

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ML-Kaggle1
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

Here is an overview of the files and directories in this repository:

*   **`Dataset/`**: Directory containing the dataset files used for training and testing the models.
    *   `training_data.csv`: Labeled data for training.
    *   `test_data.csv`: Unlabeled data for generating predictions.
    *   `example_submission.csv`: Sample format for the Kaggle submission.
*   **Notebooks (`.ipynb`)**:
    *   `data_preparation.ipynb`: Main notebook containing data exploration, preprocessing, and potentially the primary model workflow.
    *   `RandomForest.ipynb`: Implementation and tuning of the Random Forest model.
    *   `Decision Tree.ipynb`: Implementation of Decision Tree models.
    *   `LIGHTGBM.ipynb`: Experiments with LightGBM gradient boosting machine.
    *   `MLP.ipynb`: Multi-Layer Perceptron (Neural Network) implementation.
    *   `svm.ipynb`: Support Vector Machine (SVM) implementation.
    *   `Stacking.ipynb`: Ensemble method combining multiple models (Stacking) to improve prediction accuracy.
*   **Documentation & Reports**:
    *   `report.pdf`: Final project report detailing methodologies, results, and conclusions.
*   **Files**:
    *   `requirements.txt`: Python dependencies required to run the notebooks.
    *   `submission.csv`: The final generated CSV file for submission to Kaggle.