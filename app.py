from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from DataCleaningPhase1 import DataCleaner
from ModelsPhase2 import Models
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)


# Created Custom DataCleaner class which is using here to load the dataset and clean it.
cleaner = DataCleaner()
job_postings_dataframe = cleaner.cleaning_data()


# Created Models class which is using here to pass the cleaned data to the models and make different predictions.
models = Models()
knn_model = models.KNNModel(job_postings_dataframe)
logistic_model = models.LogisticRegressionModel(job_postings_dataframe)
naive_bayes_model = models.NaiveBayesModel(job_postings_dataframe)


# Using DataCleaner object to trace the actual values of labeled data
work_types = cleaner.get_unique_work_types_with_codes(job_postings_dataframe)
experience_levels = cleaner.get_unique_experience_level_with_codes(job_postings_dataframe)
application_types = cleaner.get_unique_application_type_with_codes(job_postings_dataframe)


# This is a home API which returns index.html. 
@app.route('/')
def index():
    return render_template('index.html')


# This is a company portal API which returns company.html page.
@app.route('/company')
def company():
    return render_template('company.html',work_types=work_types, experience_levels = experience_levels, application_types = application_types)


# This is a student portal API which returns student.html page.
@app.route('/student')
def student():
    return render_template('student.html',work_types=work_types, experience_levels = experience_levels, application_types = application_types)


# This is a visualization API which returns a page where user can see the visualizations of our data.
@app.route('/visualization')
def visualization():

    """
        This is a post API in which  we are showing the visualizations of our dataset. This visualization will help the user 
        to use the application with all information in-hands.
        Input: Cleaned Dataset
        Output: Multiple Visualization Graphs showing relationships between different features of dataset.
    """

    # Plotting a bar graph to visualize medium salary offered across different experience levels.
    job_posting_experience = job_postings_dataframe.groupby("formatted_experience_level")[["med_salary"]].mean()
    plt.bar(job_posting_experience.index, job_posting_experience['med_salary'])
    plt.title("Average Salary based on experience level")
    plt.xlabel("Experience Level")
    plt.xticks(rotation=20)
    plt.ylabel("Average Salary")
    plt.show()
    plt.savefig('static/ExpLevelPlot.png')
    plt.close()

    # Plotting a bar graph to show the number of job postings based on experience levels
    pd.value_counts(job_postings_dataframe['formatted_experience_level']).plot.bar()
    plt.title("Job posting based on experience levels")
    plt.xlabel("Experience levels")
    plt.xticks(rotation=20)
    plt.ylabel("Count of job postings")
    plt.savefig('static/experienceLevel.png')
    plt.close()

    # Plotting a Ddistribution of remote work options
    job_postings_dataframe['remote_allowed'] = job_postings_dataframe['remote_allowed'].map({1: 'Yes', 0: 'No'})
    remote_work_distribution = job_postings_dataframe['remote_allowed'].value_counts()
    plt.figure(figsize=(12, 9))
    wedges, texts, autotexts = plt.pie(remote_work_distribution, labels=remote_work_distribution.index, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightgreen'])
    plt.setp(autotexts, size=15)
    plt.title('Distribution of Remote Jobs Allowed')
    plt.ylabel('')
    plt.savefig('static/remoteAllowed.png')
    plt.close()

    # Plotting a distribution of work types options
    work_type_distribution = job_postings_dataframe['work_type'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 9))
    wedges, texts, autotexts = ax.pie(work_type_distribution, autopct=lambda pct: "{:.1f}%".format(pct) if pct > 0 else '', 
                                    startangle=140, 
                                    colors=['lightcoral', 'lightgreen', 'lightblue', 'lightpink', 'lightgoldenrodyellow', 'lightseagreen', 'lightskyblue'])

    plt.setp(autotexts, size=15)
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        ax.annotate(work_type_distribution.index[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment,
                    arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle))
    ax.set_title('Distribution of Work Types')
    ax.set_ylabel('')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig('static/workType.png')
    plt.close()

    # Plotting a bar graph to show relationship between number of job postings opened and number of applications received
    plt.figure(figsize=(12, 7))
    sns.histplot(job_postings_dataframe['applies'], bins=50, color='lightblue', kde=True)
    plt.title('Distribution of Number of Applications for Job Postings')
    plt.xlabel('Number of Applications')
    plt.ylabel('Number of Job Postings')
    plt.xlim(0, job_postings_dataframe['applies'].quantile(0.95))
    image_path = 'static/appliesPlot.png'
    plt.savefig(image_path)
    plt.close()

    # Returning a result as visualization.html 
    return render_template('visualization.html')


# This is an API which will show page that will predict if remote jobs are available or not based on your requirements
@app.route('/remoteallowed', methods=['POST'])
def remoteAllowed():

    """
        This is a post API that accepts some features of the dataset as an input and we are using trained logistic regression 
        model to predict if remote jobs will be available or not based on the user input.
        Inputs: Minimum salary expectated, work type expected, sponsorship required, experience level user have, type of application
        Output: Remote job is available or not
    """

    try:
        # Taking input parameters entered in the form.
        form_data = request.form
        min_salary = form_data.get('min_salary')
        formatted_work_type = form_data.get('formatted_work_type')
        sponsored = form_data.get('sponsored')
        formatted_experience_level_label_encoding = form_data.get('experience_level')
        application_type_label_encoding = form_data.get('application_type')
        applies = 1
        views = 1

        # Storing input values in an array and passing it in trained logistic regression model to predict
        input_data = np.array([
            min_salary, formatted_work_type, sponsored, formatted_experience_level_label_encoding, 
            application_type_label_encoding, applies, views
        ]).astype(float).reshape(1, -1)
        prediction = logistic_model.predict(input_data)
        prediction = prediction[0].item()

        # If the predicted value is 0 then remote jobs are not available.
        # If the predicted value is 1 then remote jobs are available.
        if prediction == 0:
            prediction = "No"
        else:
            prediction = "Yes"

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


# This is an API which will show a page that will predict the work type of jobs available based on your requirements
@app.route('/worktype', methods=['POST'])
def worktype():

    """
        This is a post API in which we are predicting the work type of jobs available based on the user inputs collected from
        the form.
        Input: Minimum Salary expectations, if user wants remote job or not, sponsorship required, experience level user have, type of application
        Output: Work type available for the expected job
    """

    try:
        # Taking input parameters entered in the form.
        form_data = request.form
        min_salary = form_data.get('min_salary')
        remote_allowed = form_data.get('remote_allowed')
        sponsored = form_data.get('sponsored')
        formatted_experience_level_label_encoding = form_data.get('experience_level')
        application_type_label_encoding = form_data.get('application_type')

         # Storing input values in an array and passing it in trained naive bayes model to predict
        input_data = np.array([
            min_salary, remote_allowed, sponsored, application_type_label_encoding, 
            formatted_experience_level_label_encoding
        ]).astype(float).reshape(1, -1)
        prediction = naive_bayes_model.predict(input_data)
        prediction = prediction[0].item()
        # print(prediction)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


# This is an API which will redirect to a page that will predict the maximum salary range that comapny can offer based on your requirements
@app.route('/predict', methods=['POST'])
def predict():

    """
        This is a post API which is predicting the maximum salary range that a company can offer to the employee based on the 
        user inputs of job requirements.
        Input: Minimum salary that company can offer, work type company expects, if job is remote or not, looking for which experience level, type of application
        Output: Maximum salary range that comapny can offer
    """

    try:
        # Taking input parameters entered in the form.
        form_data = request.form
        min_salary = form_data.get('min_salary')
        formatted_work_type = form_data.get('formatted_work_type')
        remote_allowed = form_data.get('remote_allowed')
        formatted_experience_level_label_encoding = form_data.get('experience_level')
        application_type_label_encoding = form_data.get('application_type')

        # Storing input values in an array and passing it in trained KNN model to predict
        input_data = np.array([
            min_salary, formatted_work_type, remote_allowed, formatted_experience_level_label_encoding,
            application_type_label_encoding
        ]).astype(float).reshape(1, -1)
        prediction = knn_model.predict(input_data)
        prediction_range = models.map_bin_to_range(int(prediction[0]))
        
        # Returning result as prediction_max_salary.html
        return render_template('prediction_max_salary.html', prediction=prediction_range)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
