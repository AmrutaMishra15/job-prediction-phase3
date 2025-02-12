# Job Prediction Service Application

Job Prediction Service App-  Final Application to predict Job availability statistics as well as through analysis regarding job availability, job Type, Sponsorship and other crucial information that helps the labor market to make efficient decisions in the rapidly evolving labor market.

● In Phase 1 of the project we performed EDA to explore the data, and formulated hypotheses that led to new data collection and experiments. Also, performed various types of Visualization like correlation matrix, Relation between job viewed and applied, job postings based on experience level, Average salary based on experience level and remote job distribution etc., helped to understand the relationship between numerical columns in the dataset.

● In Phase 2 of the project after EDA, we applied significant algorithms from Machine Learning (ML) like Linear regression, Logistic Regression, SVM, KNN, Random Forest, Naive Bayes to gain intelligence from data. we used various evaluation metrics like R2_score, accuracy_score, mean_squared_error, mean_absolute_error to evaluate the model accurately.

● After careful evaluation of all algorithms we finalized the model that predicted more accurately with better score than other models. The Models finalized were Logistic Regression, Naive Bayes and KNN to make predictions like max salary range, remote jobs allowed or not and the work types based on user inputs along with our dataset.

● Finally Designed an application using Flask, HTML and Python that give statistics on Job market as well as taken required inputs, processed through the model and predicted the results.It address issues like:

1. whether or not remote jobs are allowed or not based on certain inputs like experience level, minimum salary expected, application type, sponsorship etc.
2. one can also get insights about the current job trends like what is the remote job distributions, Various types of work and its percentage of distribution, Average salary offered by the companies for job profiles. It can also be used by companies and researchers to look at the job market trends
3. Job Seekers can also predict the possibility of getting remote jobs based on their experience levels, Application type, sponsorship etc. They can also predict the possible work type they should be looking for based on their profile, expectation and the market trends.
4. HR professionals of companies can also make use of the analysis to tailor their job offerings like what Maximum salary bracket that should be offered against other industries.
