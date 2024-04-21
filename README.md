# Travel-Accommodation-Prediction



Goal :The goal of the project is to predict the accommodation type for a trip based on various traveler and trip features.


## Introduction
This project aims to predict the accommodation type (apartment/hotel) for trips based on various traveler and trip features. The dataset used for this project is sourced from a travel agency and contains seven attributes: "id", "durationOfStay", "gender", "Age", "kids", "destinationCode", and "AcomType."

## Dataset
The dataset is provided in a TXT file named `train_data.txt`. It contains one record per line, with each record representing a trip and its attributes in a specific order:
- "id": Unique trip ID
- "durationOfStay": Duration of the trip in days
- "gender": Gender of the booker (F for female, M for male)
- "Age": Age of the booker in years
- "kids": Whether there are kids in the travel party (0 for false, 1 for true)
- "destinationCode": Destination country code
- "AcomType": Accommodation type (apartment/hotel)

## Project Workflow
The project follows the following workflow:
1. Data Cleaning: The data is read from the `train_data.txt` file and cleaned to remove "[1]" and redundant spaces. Missing values are replaced with "NaN."
2. Data Transformation: The data is reshaped from a single column to a DataFrame with eight columns for ease of analysis.
3. Data Split: The data is split into features (X) and the target variable (y). Further, it is horizontally split into training and testing sets.
4. Model Selection: A RandomForestClassifier is selected as the predictive model due to its suitability for classification tasks.
5. Model Training and Evaluation: The model is trained on the training set and evaluated using the test set. GridSearchCV is used for hyperparameter tuning.

## Data Exploration
In this project, we performed data exploration to gain insights into the dataset and better understand the relationships between different features.

### Class Balance
We first checked the class balance of the target variable, "AcomType," by creating a bar plot showing the frequency of each class (apartment/hotel). This plot helps us understand whether the dataset is balanced or imbalanced.

### Dealing with Missing Values
We addressed missing values in the "Age" column by filling them with the median age. The missing values in the "destinationCode" column were filled with the most frequent destination code.

### Feature Visualization
We created several visualizations to understand the distribution and relationships between features.

- Age Distribution: We created a histogram to visualize the distribution of ages in the dataset.
- Duration of Stay Distribution: A histogram was used to show the distribution of trip durations.
- Relationship between Age and Accommodation Type: A box plot was generated to explore the relationship between age and the accommodation type (apartment/hotel).
- Gender and Accommodation Type: Stacked bar charts were used to display the relationship between gender and accommodation type.
- Kids and Accommodation Type: Stacked bar charts were used to examine the relationship between having kids and accommodation type.
- Destination and Accommodation Type: Stacked bar charts were used to analyze the relationship between destination and accommodation type.

### Feature Correlation
We computed the correlation matrix for numerical features to explore any potential relationships between them. A heatmap was created to visualize the correlations.

For detailed code implementation and visualizations, please refer to the Jupyter notebook or Python script provided in this repository.

## Installation and Usage
1. Clone the repository to your local machine.
2. Make sure you have Python and the required libraries installed (listed in `requirements.txt`).
3. Run the Jupyter notebook or Python script to execute the project.
4. The notebook/script will perform data preprocessing, model training, and evaluation, and display the training and test accuracies.

## Dependencies
The project requires the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- category_encoders

You can install them using pip or any package manager by running:

## Results
The model's performance is evaluated based on the accuracy metric. Training accuracy and test accuracy are computed and displayed at the end of the notebook/script.

## Future Improvements
While the current implementation provides a baseline model for accommodation prediction, further improvements can be made. Some potential areas of enhancement include:
- Feature engineering: Creating additional relevant features for better prediction.
- Trying different classification algorithms to see if there is a better-performing model.
- Exploring more advanced hyperparameter tuning techniques.
- Conducting deeper exploratory data analysis for additional insights.

## Contact Information
If you have any questions or feedback regarding this project, feel free to contact the project owner:
- Name: [Maryam Ramezani]
- Email: [ram.mar.math@gmail.com]
- GitHub: [https://github.com/MaryamRamezaniGithub/QualogyDataTest]

