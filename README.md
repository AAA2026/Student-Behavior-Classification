## Students Behavior Analysis

This project analyzes a comprehensive dataset of student academic and behavioral information with the primary goal of classifying student clickstream patterns as either 'normal' or 'unusual'. The analysis encompasses several stages, including data loading, thorough preprocessing, feature engineering, training machine learning models, and evaluating their performance. The core classification workflow is detailed in the `PR_ML_idea2.ipynb` notebook, which builds upon initial data exploration and cleaning steps performed in `PR_ML_idea1.ipynb`.

### Dataset

### The analysis utilizes the `enhanced_dataset.csv` file, which contains 105,645 records, each representing a student instance with 29 distinct features. This rich dataset includes a variety of information categories:

*   **Demographics:** Features such as gender, major, country of origin, and academic year level.
*   **Academic Performance:** Metrics like Grade Point Average (GPA), midterm scores, average scores on assignments and quizzes, previous course scores, and attendance percentages.
*   **Study Habits and Well-being:** Data on weekly study hours, nightly sleep hours, participation scores, and reported stress levels.
*   **Online Activity:** Indicators related to online learning behavior, including the device used for quizzes, time spent on tasks, observed clickstream patterns ('normal' or 'unusual'), and the type of online activity engaged in (e.g., online courses, forums, study groups).

### Data Preprocessing and Feature Engineering

The project involved a multi-step preprocessing pipeline to prepare the data for modeling:

1.  **Initial Exploration (`PR_ML_idea1.ipynb`):** This phase involved loading the dataset, examining feature data types, and identifying missing values, which were noted in columns like `quiz_id`, `duration_limit`, and `question_count`. Feature selection was performed, initially focusing on numerical features potentially predictive of `prev_score`. Outlier detection and removal using the Interquartile Range (IQR) method were explored for these academic score features.

2.  **Classification-Specific Preprocessing (`PR_ML_idea2.ipynb`):** The main workflow focused on the clickstream classification task. Key steps included:
    *   **Categorical Feature Encoding:** Features like 'gender', 'major', 'year_level', 'clickstream' (the target variable), and 'activity' were converted into numerical representations using Label Encoding.
    *   **Feature Scaling:** Relevant numerical features selected for the classification model were standardized using `StandardScaler`. This ensures that features with larger value ranges do not disproportionately influence the model's learning process.
    *   **Target Variable Preparation:** The 'clickstream' column was encoded into numerical labels suitable for the classification algorithms.

### Modeling

After preprocessing, the dataset was divided into training (80% of the data) and testing (20%) sets to facilitate model training and unbiased evaluation. Two distinct classification algorithms were employed:

*   **Random Forest Classifier:** This ensemble learning method builds multiple decision trees during training and outputs the class that is the mode of the classes output by individual trees. It is known for its high accuracy, robustness to overfitting, and ability to handle complex interactions between features.
*   **Support Vector Classifier (SVC):** A powerful and versatile classification model that finds an optimal hyperplane to separate different classes in the feature space. It is particularly effective in high-dimensional spaces and when the number of dimensions exceeds the number of samples.

### Evaluation

### The performance of the trained Random Forest and SVC models was rigorously assessed on the unseen test set using several standard classification metrics:

*   **Accuracy:** Both models demonstrated exceptionally high predictive accuracy, achieving approximately 99.99%.
*   **Classification Report:** This provided a detailed breakdown of performance, including precision, recall, and F1-score for each class ('normal' and 'unusual'). These metrics offer insights into the models' ability to correctly identify each specific clickstream pattern.
*   **Confusion Matrix:** A confusion matrix was generated for each model to visualize the relationship between predicted and actual class labels. The matrices confirmed the high overall accuracy, showing very few instances of misclassification.

### How to Run

To replicate this analysis, you will need a Python environment with the following core libraries installed: `pandas` (for data manipulation), `matplotlib` and `seaborn` (for visualization), and `scikit-learn` (for machine learning tasks including preprocessing, modeling, and evaluation).

1.  Ensure the `enhanced_dataset.csv` file is accessible in the path specified within the notebooks, or update the file path accordingly.
   
2.  Execute the Jupyter notebooks:
    *   Run `PR_ML_idea1.ipynb` for the initial data loading, exploration, and preliminary cleaning steps.
    *   Run `PR_ML_idea2.ipynb` to perform the main classification workflow, including specific preprocessing, feature engineering, model training (Random Forest and SVC), and performance evaluation.

### Notebook Descriptions

*   **`PR_ML_idea1.ipynb`:** Focuses on loading the dataset, conducting initial exploratory data analysis (EDA), handling basic data cleaning, and setting up features potentially relevant for a regression task (predicting `prev_score`), including outlier analysis.
*   **`PR_ML_idea2.ipynb`:** Implements the primary machine learning task of classifying student 'clickstream' behavior. This includes loading data, performing preprocessing steps tailored for classification (encoding categorical features, scaling numerical ones), splitting data, training Random Forest and Support Vector Classifier models, and evaluating their performance using accuracy, classification reports, and confusion matrices.
