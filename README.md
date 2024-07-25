# Aviakompaniya Passengers Satisfaction

## Overview
This project aims to analyze passenger satisfaction for an airline company using various data analysis and machine learning techniques. The analysis and model training were performed using popular Python libraries and classifiers. The best-performing model was used to train the data for predicting passenger satisfaction.

## Technologies and Libraries
- **Data Analysis**:
  - **Numpy**: For numerical computations.
  - **Pandas**: For data manipulation and analysis.
  - **matplotlib.pyplot**: For data visualization.
  - **seaborn**: For statistical data visualization.
  
- **Model Training**:
  - **KNeighborsClassifier**: For k-nearest neighbors classification.
  - **DecisionTreeClassifier**: For decision tree classification.
  - **RandomForestClassifier**: For random forest classification.
  - **LogisticRegression**: For logistic regression classification.

## Training Results
The performance of various classifiers was evaluated, and the results are as follows:

| Classifier              | Dataset  | Train-set Score | Test-set Score | Satisfied Accuracy | Dissatisfied Accuracy |
|-------------------------|----------|-----------------|----------------|--------------------|-----------------------|
| DecisionTreeClassifier  | Original | 98.08%          | 93.18%         | 94.27%             | 92.06%                |
| KNeighborsClassifier    | Original | 92.89%          | 90.58%         | 94.87%             | 86.15%                |
| LogisticRegression      | Original | 87.03%          | 86.97%         | 89.14%             | 84.73%                |
| RandomForestClassifier  | Original | 99.03%          | 94.59%         | 95.76%             | 93.38%                |

The **RandomForestClassifier** was selected as the best model based on its superior performance in accuracy, precision, recall, and F1 score.

## Project Structure
- **AviakompaniyaPassengersSatisfaction.ipynb**: The main Jupyter notebook file containing the entire project including data analysis, model training, and evaluation.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or Google Colab
- pip (Python package installer)

### Running the Project in Google Colab
1. Open the GitHub repository containing the notebook:
    ```bash
    https://github.com/dostonshernazarov/Passenger-satisfaction-predict-model
    ```

2. Open the `AviakompaniyaPassengersSatisfaction.ipynb` file in Google Colab by clicking on the "Open in Colab" button or using the following link:
    [Open in Colab](https://colab.research.google.com/github/dostonshernazarov/Passenger-satisfaction-predict-model/blob/main/AviakompaniyaPassengersSatisfaction.ipynb)

3. Run all cells in the notebook to perform data analysis and model training.

### Running the Project Locally
1. Clone the repository:
    ```bash
    git clone https://github.com/dostonshernazarov/Passenger-satisfaction-predict-model.git
    cd Passenger-satisfaction-predict-model
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter notebook:
    ```bash
    jupyter notebook AviakompaniyaPassengersSatisfaction.ipynb
    ```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements
- Thanks to the developers of the libraries used in this project.

## Contact
For any inquiries or questions, please contact [dostonshernazarov989@gmail.com].
