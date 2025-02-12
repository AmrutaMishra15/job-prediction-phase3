from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd

class Models:

    # Initializing salary bins and labels in the constructor.
    def __init__(self) -> None:
        self.salary_bins = [0, 20000, 60000, 100000, 200000, 600000, 1000000, 1500000]
        self.salary_labels = [1, 2, 3, 4, 5, 6, 7]
        pass

    
    # Training the KNN Model and returning trained model
    def KNNModel(self, dataframe):

        """
            In this function we are taking relevant features from the dataset and training the model using KNN Algorithm to 
            predict maximum salary range that company can offer.
            Input: Minimum salary that company can offer, work type company expects, if job is remote or not, looking for which experience level, type of application
            Output: Trained KNN Model
        """

        # Selecting X features as an input and Y target feature as an output
        selected_features = ['min_salary', 'formatted_work_type', 'remote_allowed', 'formatted_experience_level_label_encoding', 'application_type_label_encoding']
        X_data = dataframe[selected_features]
        dataframe['max_salary_range'] = pd.cut(dataframe['max_salary'], bins=self.salary_bins, labels=self.salary_labels)
        Y_target_knn = dataframe['max_salary_range']

        # Splitting the data into 80:20 ratio to train and test the model.
        X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_data, Y_target_knn, test_size=0.2, random_state=0)

        # Performing train, fit and predict operations on the database with KNN classifier.
        knn_model = KNeighborsClassifier(n_neighbors=7)
        knn_model.fit(X_train_knn, y_train_knn)
        y_predicted_knn = knn_model.predict(X_test_knn)
        # print(y_predicted_knn)

        # Checking the accuracy of the model and printing it.
        knn_accuracy = accuracy_score(y_test_knn, y_predicted_knn)
        print("KNN Model Accuracy: ", knn_accuracy*100)

        # Returning trained KNN model
        return knn_model
    

    # Checking the range of bracket based on bins
    def map_bin_to_range(self, bin_number):

        """
            This function takes bin number as an input and check the bin bracket.
            Input: bin number
            Output: bin bracket
        """

        # checking the base condition and then returning the bin bracket based on the bin number
        if bin_number < 1 or bin_number > len(self.salary_bins) - 1:
            return "Unknown range"
        return f"{self.salary_bins[bin_number - 1]} - {self.salary_bins[bin_number]}"
    

    # Training the Logistic Regression Model and returning trained model
    def LogisticRegressionModel(self, dataframe):

        """
            In this function we are taking relevant features from the dataset and training the model using Logistic Regression 
            Algorithm to predict availability of remote jobs.
            Inputs: Minimum salary expectated, work type expected, sponsorship required, experience level user have, type of application
            Output: Trained Logistic Model
        """

        # Selecting X features as an input and Y target feature as an output
        selected_columns = ['min_salary', 'formatted_work_type', 'sponsored', 'formatted_experience_level_label_encoding', 'application_type_label_encoding', 'applies', 'views']
        X_data_selected_column = dataframe[selected_columns]
        Y_target_logistic = dataframe['remote_allowed']

         # Splitting the data into 80:20 ratio to train and test the model.
        X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_data_selected_column, Y_target_logistic, test_size=0.2, random_state=0)

         # Performing train, fit and predict operations on the database with logistic regression model.
        logistic_regression_model = LogisticRegression(max_iter=500)
        logistic_regression_model.fit(X_train_logistic, y_train_logistic)
        y_predicted_logistic = logistic_regression_model.predict(X_test_logistic)
        # print(y_predicted_logistic)

        # Checking the accuracy of the model and printing it.
        logistic_regression_accuracy = accuracy_score(y_test_logistic, y_predicted_logistic)
        print("Logistic Regression Accuracy: ", logistic_regression_accuracy*100)

        # Returning trained Logistic Regression model
        return logistic_regression_model
    

    # Training the Naive Bayes Model and returning trained model
    def NaiveBayesModel(self, dataframe):
        
        """
            In this function we are taking relevant features from the dataset and training the model using Naive Bayes 
            Algorithm to predict work type of available jobs.
            Input: Minimum Salary expectations, if user wants remote job or not, sponsorship required, experience level user have, type of application
            Output: Trained Naive Bayes Model
        """

        # Selecting X features as an input and Y target feature as an output
        selected_columns_nb = ['min_salary', 'remote_allowed', 'sponsored', 'application_type_label_encoding', 'formatted_experience_level_label_encoding']
        X_data_nb = dataframe[selected_columns_nb]
        Y_target_nb = dataframe['work_type']

         # Splitting the data into 80:20 ratio to train and test the model.
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_data_nb, Y_target_nb, test_size=0.2, random_state=0)

         # Performing train, fit and predict operations on the database with naive bayes model.
        naive_bayes_model = GaussianNB()
        naive_bayes_model.fit(X_train_nb, y_train_nb)
        y_predicted_nb =  naive_bayes_model.predict(X_test_nb)
        # print(y_predicted_nb)

        # Checking the accuracy of the model and printing it.
        naive_bayes_accuracy = accuracy_score(y_test_nb, y_predicted_nb)
        print("Naive Bayes Accuracy: ", naive_bayes_accuracy*100)

        # Returning trained Naive Bayes model
        return naive_bayes_model


