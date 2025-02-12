import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class DataCleaner:

    def __init__(self) -> None:
        pass


    # data type conversion from number to date time  for Data time fields
    def convert_to_datetime(self, dataFrame, columnName):

        """
            In this function we are taking the dataframe and column name as an input and converting the values in that
            column to date time structure
            Input: Dataframe, Column Name
            Output: Changed Dataframe with updated values in the given column
        """

        # Using to_datetime() function of pandas to convert the float value into date time format.
        dataFrame[columnName] = dataFrame[columnName].astype(float)
        dataFrame[columnName] = pd.to_datetime(dataFrame[columnName], unit='ms')


    # Data Transformation/ Normalization (the max_salary ,min_salary and med_salary based on the Pay period)
    def normalize_pay_period(self, dataframe, columnName):

        """
            In this function we are normalizing the salary based on pay period as they have been declared as 'HOURLY', 'MONTHLY'
            and 'YEARLY' in the column pay_period.
            Input: Dataframe, Column Name
            Output: Modified dataset with updated values in the given column
        """

        yearly = 1
        monthly = 12
        hourly = 40*52

        # Changing the values in column based on the values in the pay_period column
        dataframe[columnName] = dataframe.apply(lambda x: x[columnName] * hourly if x['pay_period'] == ('HOURLY')
                                                else (x[columnName] * monthly if x['pay_period'] == ('MONTHLY')
                                                else x[columnName] * yearly ), axis=1)


    # Filling empty columns of experience level based on title
    def fill_experience_level_title(self, job_title):

        """
            In this function we are returning the experience level values based on the title of the job. There are some rows in 
            the dataset which don't have exeperinece level value filled. So, this function will fill those rows.
            Input: Job Title
            Output: Experience Level based on values in the job title
        """

        if 'director' in job_title.lower():
            return 'DIRECTOR'
        elif 'manager' in job_title.lower():
            return 'MID-SENIOR LEVEL'
        elif 'senior' in job_title.lower() or 'specialist' in job_title.lower() or 'sr' in job_title.lower():
            return 'SENIOR LEVEL'
        elif 'associate' in job_title.lower() or 'I' in job_title.lower() or 'clerk' in job_title.lower() or 'assistant' in job_title.lower() or 'jr' in job_title.lower()or 'analyst' in job_title.lower() or 'junior' in job_title.lower() or 'administrator' in job_title.lower():
            return 'ENTRY LEVEL'
        elif 'intern' in job_title.lower():
            return 'INTERNSHIP'
        elif 'owner' in job_title.lower():
            return 'CEO'
        elif 'executive' in job_title.lower():
            return 'EXECUTIVE'
        
    
    # Filling empty columns of experience level based on description
    def fill_experience_level_description(self, description):

        """
            In this function we are returning the experience level values based on the description of the job. There are some 
            rows in the dataset which don't have exeperinece level value filled. So, this function will fill those rows.
            Input: Job Description
            Output: Experience Level based on values in the job description
        """

        if 'director' in description.lower():
            return 'DIRECTOR'
        elif 'manager' in description.lower() or 'III' in description.lower():
            return 'MID-SENIOR LEVEL'
        elif 'senior' in description.lower() or 'specialist' in description.lower() or 'sr' in description.lower() or 'II' in description.lower():
            return 'SENIOR LEVEL'
        elif 'associate' in description.lower() or 'I' in description.lower() or 'clerk' in description.lower() or 'assistant' in description.lower() or 'jr' in description.lower()or 'analyst' in description.lower() or 'junior' in description.lower() or 'administrator' in description.lower():
            return 'ENTRY LEVEL'
        elif 'intern' in description.lower():
            return 'INTERNSHIP'
        elif 'owner' in description.lower():
            return 'CEO'
        elif 'executive' in description.lower():
            return 'EXECUTIVE'
        

    # Filling empty columns of experience level based on salary
    def fill_experience_level_salary(self, salary):

        """
            In this function we are returning the experience level values based on the description of the job. There are some 
            rows in the dataset which don't have exeperinece level value filled. So, this function will fill those rows.
            Input: Salary
            Output: Experience Level based on values in the salary
        """

        if salary > 100000:
            return 'MID-SENIOR LEVEL'
        elif salary > 60000 and salary <= 100000:
            return 'SENIOR LEVEL'
        elif salary <= 60000:
            return 'ENTRY LEVEL'

    # Applying multiple data cleaning methods and returning the cleaned datafram
    def cleaning_data(self):

        """
            In this function, we are reading values from our dataset and applying data cleaning and exploratory data analysis 
            methods on that dataset. Returning the cleaned dataframe.
            Input: Original Dataset
            Output: Cleaned Dataframe
        """

        # fetching the data and creating dataFrame
        job_postings_dataset = pd.read_csv("job_postings.csv")
        job_postings_dataframe = pd.DataFrame(job_postings_dataset)

        # checking the information of dataframe
        job_postings_dataframe.info()

        #  finding the number of fetaures and rows of the data
        job_postings_dataframe.shape

        #  finding the statistic of the given dataset for better analysis
        job_postings_dataframe.describe()

        #  finding null value in each of the features in dataset
        missing_values = job_postings_dataframe.isnull()
        for columns in missing_values.columns.values.tolist():
            print(missing_values[columns].value_counts(),end="\n")

        #  Drop duplicates values if present and keep the lastest value
        job_postings_dataframe.drop_duplicates(keep='last', inplace=True)

        # fill the columns like applies or views with 0 if blank for handling missing data 
        job_postings_dataframe['applies'].fillna(0, inplace=True)
        job_postings_dataframe['views'].fillna(0, inplace=True)
        job_postings_dataframe['remote_allowed'].fillna(0, inplace=True)

        # Text formatting for string type columns to make data consistent
        job_postings_dataframe.loc[:, 'formatted_work_type'] = job_postings_dataframe.loc[:, 'formatted_work_type'].str.upper()
        job_postings_dataframe.loc[:, 'work_type'] = job_postings_dataframe.loc[:, 'work_type'].str.upper()
        job_postings_dataframe.loc[:, 'formatted_experience_level'] = job_postings_dataframe.loc[:, 'formatted_experience_level'].str.upper()
        job_postings_dataframe.loc[:, 'application_type'] = job_postings_dataframe.loc[:, 'application_type'].str.upper()

        # Encode categorical values using Label encoding
        job_postings_dataframe.loc[:, 'formatted_work_type'] = job_postings_dataframe.loc[:, 'formatted_work_type'].astype('category').cat.codes
        job_postings_dataframe.loc[:, 'job_posting_url'] = job_postings_dataframe.loc[:, 'job_posting_url'].astype('category').cat.codes
        job_postings_dataframe.loc[:, 'application_url'] = job_postings_dataframe.loc[:, 'application_url'].astype('category').cat.codes
        job_postings_dataframe.loc[:, 'posting_domain'] = job_postings_dataframe.loc[:, 'posting_domain'].astype('category').cat.codes

        # Converting the columns which have dates in their entry but stored as a string
        self.convert_to_datetime(job_postings_dataframe, 'expiry')
        self.convert_to_datetime(job_postings_dataframe, 'original_listed_time')
        self.convert_to_datetime(job_postings_dataframe, 'listed_time')
        self.convert_to_datetime(job_postings_dataframe, 'closed_time')

        # Dropping null values (max_salary and min_salary are important to predict , so dropping the null values as they are irrelevant)
        job_postings_dataframe.dropna(subset=['max_salary', 'min_salary'], axis=0, inplace=True)

        # Inconsistent data handling (Finding  median salary fron max_salary and min_salary where it is absent)
        job_postings_dataframe["med_salary"] = job_postings_dataframe.apply(lambda x: 
            (x["max_salary"] + x["min_salary"]) / 2 if x["max_salary"] and x["min_salary"] is not None 
            else x["med_salary"], axis=1)
        
        # Normalizing the salary based on their pay period
        self.normalize_pay_period(job_postings_dataframe, 'max_salary')
        self.normalize_pay_period(job_postings_dataframe, 'min_salary')
        self.normalize_pay_period(job_postings_dataframe, 'med_salary')
        job_postings_dataframe['pay_period'] =  'YEARLY'

        # Filling the empty cells of formatted_experience_level based on the job title
        job_postings_dataframe['formatted_experience_level'] = job_postings_dataframe.apply(
            lambda row: self.fill_experience_level_title(row['title']) if pd.isna(row['formatted_experience_level']) else row['formatted_experience_level'],
            axis=1
        )

        # Filling the empty cells of formatted_experience_level based on the job description
        job_postings_dataframe['formatted_experience_level'] = job_postings_dataframe.apply(
            lambda row: self.fill_experience_level_description(row['description']) if pd.isna(row['formatted_experience_level']) else row['formatted_experience_level'],
            axis=1
        )

        # Filling the empty cells of formatted_experience_level based on the salary
        job_postings_dataframe['formatted_experience_level'] = job_postings_dataframe.apply(
            lambda row: self.fill_experience_level_salary(row['min_salary']) if pd.isna(row['formatted_experience_level']) else row['formatted_experience_level'],
            axis=1
        )

        # Encoding categorical values using Label encoding
        job_postings_dataframe.loc[:, 'formatted_experience_level_label_encoding'] = job_postings_dataframe.loc[:, 'formatted_experience_level'].astype('category').cat.codes
        job_postings_dataframe.loc[:, 'application_type_label_encoding'] = job_postings_dataframe.loc[:, 'application_type'].astype('category').cat.codes

        # creating a new csv file for cleaned data
        job_postings_dataframe.to_csv("job_postings_cleaned.csv", index=False)

        return job_postings_dataframe


    def get_unique_work_types_with_codes(self,dataframe):

        """
            In this function, we are reading unique Work Type values from our dataset and creatiing dictionary 
            with the work type labels as key and it's respective Categorical codes as value. 
            It is a helper fucntion that returns the work type List which is used in html forms to load dropdown options. 
            Input: Original Dataset
            Output: Work Type List
        """
        label_encoder = LabelEncoder()
        dataframe['work_type_code'] = label_encoder.fit_transform(dataframe['work_type'])
        work_type_codes = list(zip(dataframe['work_type'], dataframe['work_type_code']))
        work_type_codes = list(set(work_type_codes))
        work_type_codes.sort(key=lambda x: x[1])
        return work_type_codes
    
    def get_unique_experience_level_with_codes(self,dataframe):
        """
            In this function, we are reading unique Experience Level values from our dataset and creatiing dictionary 
            with the Experience Level labels as key and it's respective Categorical codes as value. 
            It is a helper fucntion that returns the Experience Level List which is used in html forms to load dropdown options. 
            Input: Original Dataset
            Output: Experience Level List
        """
        label_encoder = LabelEncoder()
        dataframe['formatted_experience_level_code'] = label_encoder.fit_transform(dataframe['formatted_experience_level'])
        formatted_experience_level_code = list(zip(dataframe['formatted_experience_level'], dataframe['formatted_experience_level_code']))
        formatted_experience_level_code = list(set(formatted_experience_level_code))
        formatted_experience_level_code.sort(key=lambda x: x[1])
        return formatted_experience_level_code
    
    def get_unique_application_type_with_codes(self,dataframe):
        """
            In this function, we are reading unique Application type values from our dataset and creatiing dictionary 
            with the Application type labels as key and it's respective Categorical codes as value. 
            It is a helper fucntion that returns the Application type List which is used in html forms to load dropdown options. 
            Input: Original Dataset
            Output: Application type List
        """
        label_encoder = LabelEncoder()
        dataframe['application_type_level_code'] = label_encoder.fit_transform(dataframe['application_type'])
        application_type_level_code = list(zip(dataframe['application_type'], dataframe['application_type_level_code']))
        application_type_level_code = list(set(application_type_level_code))
        application_type_level_code.sort(key=lambda x: x[1])
        return application_type_level_code

