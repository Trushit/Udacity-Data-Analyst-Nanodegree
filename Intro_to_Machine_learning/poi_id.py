# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Creating pandas dataframe
import pandas as pd
enron = pd.DataFrame(data_dict)
enron = enron.transpose()

#Converting columns to appropriate datatype

enron = enron.astype(dtype= {"bonus":"float64","deferral_payments":"float64",
                             "deferred_income":"float64","director_fees":"float64",
                             "exercised_stock_options":"float64","expenses":"float64",
                            "from_messages":"float64","from_poi_to_this_person":"float64",
                             "from_this_person_to_poi":"float64","loan_advances":"float64",
                             "long_term_incentive":"float64","other":"float","poi":"bool",
                             "restricted_stock":"float64","restricted_stock_deferred":"float64",
                             "salary":"float64","shared_receipt_with_poi":"float64",
                             "to_messages":"float64","total_payments":"float64",
                             "total_stock_value":"float64"})


# Summary of dataset
enron.info()
enron.describe()

#Creation of list called payment_data, stock_data and email_data
payment_data = ["bonus","deferral_payments",
                "deferred_income","director_fees",
                "expenses","loan_advances",
                "long_term_incentive","other",
                "salary","total_payments"]

stock_data = ["exercised_stock_options","restricted_stock",
              "restricted_stock_deferred","total_stock_value"]

email_data=["from_messages","from_poi_to_this_person",
            "from_this_person_to_poi","shared_receipt_with_poi",
            "to_messages"]

# Replacing NaN in financial data with zero 
enron[payment_data] = enron[payment_data].fillna(0)
enron[stock_data] = enron[stock_data].fillna(0)
 
# Replacing NaN in email data with mean of the column
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

#Filters data into POI and Non-POI
enron_poi = enron[enron['poi'] == True]
enron_nonpoi = enron[enron['poi']==False]

enron_poi.loc[:, email_data] = imp.fit_transform(enron_poi.loc[:,email_data])
enron_nonpoi.loc[:, email_data] = imp.fit_transform(enron_nonpoi.loc[:,email_data])
enron = enron_poi.append(enron_nonpoi)

#Check if total_payments is equal to total of all the payments
errors = (enron[enron[payment_data[:-1]].sum(axis='columns') != enron['total_payments']])
errors

#Correction of values for wrong entries
enron.at["BELFER ROBERT",payment_data] = [0,0,0,0,102500,0,-102500,3285,0,3285]
enron.at["BELFER ROBERT",stock_data] = [0,44093,-44093,0]

enron.at["BHATNAGAR SANJAY",payment_data] = [0,0,0,0,137864,0,0,0,0,137864]
enron.at["BHATNAGAR SANJAY",stock_data] = [15456290,2604490,-2604490,15456290]

#Check if there are any errors left

len(enron[enron[payment_data[:-1]].sum(axis='columns') != enron['total_payments']])
len(enron[enron[stock_data[:-1]].sum(axis='columns') != enron['total_stock_value']])

#Drop the variables
enron.drop(["TOTAL","THE TRAVEL AGENCY IN THE PARK"],inplace=True)

#Outlier detection

IQR = enron.quantile(q=0.75) - enron.quantile(q=0.25)
first_quartile = enron.quantile(q=0.25)
third_quartile = enron.quantile(q=0.75)
outliers = enron[(enron>(third_quartile + 1.5*IQR) ) | (enron<(first_quartile - 1.5*IQR) )].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers

# Remove the outlier individuals
enron.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C','LOCKHART EUGENE E'], inplace=True)
# Find the number of poi and non poi now in the data
enron['poi'].value_counts()

#Making poi first feature in the dataset
enron=enron[["poi","bonus","deferral_payments","deferred_income",
            "director_fees","exercised_stock_options","expenses",
             "from_messages","from_poi_to_this_person",
             "from_this_person_to_poi","loan_advances","long_term_incentive",
             "other","restricted_stock","restricted_stock_deferred",
             "salary","shared_receipt_with_poi","to_messages",
             "total_payments","total_stock_value"]]

# Add new email features to the dataframe
enron['to_poi_ratio'] = enron['from_poi_to_this_person'] / enron['to_messages']
enron['from_poi_ratio'] = enron['from_this_person_to_poi'] / enron['from_messages']
enron['shared_poi_ratio'] = enron['shared_receipt_with_poi'] / enron['to_messages']

# Create new financial features and add to the dataframe
enron['bonus_to_salary'] = enron['bonus'] / enron['salary']
enron['bonus_to_total'] = enron['bonus'] / enron['total_payments']

features_list=["poi","bonus","deferral_payments","deferred_income",
                "director_fees","exercised_stock_options","expenses",
               "from_messages","from_poi_to_this_person",
               "from_this_person_to_poi","loan_advances",
               "long_term_incentive","other","restricted_stock",
               "restricted_stock_deferred","salary","shared_receipt_with_poi",
               "to_messages","total_payments","total_stock_value",'to_poi_ratio',
               'from_poi_ratio','shared_poi_ratio','bonus_to_salary','bonus_to_total']

import numpy as np
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tester
from feature_format import featureFormat, targetFeatureSplit

enron=enron.replace([np.inf, -np.inf], np.nan)
enron=enron.fillna(0)


# Scale the dataset, convert back to dictionary. email_address is dropped.
#enron.drop('email_address', axis=2,inplace=True)
scaled_enron = enron.copy()
scaled_enron.iloc[:,1:] = scale(scaled_enron.iloc[:,1:])
my_dataset = scaled_enron.to_dict(orient='index')

# Create and test the Gaussian Naive Bayes Classifier
clf = GaussianNB()
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

# Create and test the Decision Tree Classifier
clf = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main();

# Create and test the Support Vector Classifier
clf = SVC(kernel='linear')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

#Decision Tree classifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
features_train, features_test, labels_train, labels_test = \
    train_test_split(enron.drop("poi",axis=1),enron.loc[:,"poi"],test_size=0.3, random_state=42)

#Fits the decision tree and performs prediction
clf = DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

#Evaluation of classifier
print(clf.score(features_test,labels_test))
print metrics.confusion_matrix(labels_test, pred)
print metrics.classification_report(labels_test, pred)

#Extracts feature importances
clf.feature_importances_

#Determination of number of optimal features
from sklearn.model_selection import GridSearchCV
n_features = np.arange(1, len(features_list))

# Create a pipeline with feature selection and classification
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

pipe = Pipeline([
    ('select_features', SelectKBest()),
    ('classify', DecisionTreeClassifier())
])

parameters_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
tree_clf= GridSearchCV(pipe, param_grid=parameters_grid, scoring='f1', cv = 10)
tree_clf.fit(features, labels)
tree_clf.best_params_

#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
# Create a pipeline with feature selection and classifier
pipe_tree = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier()),
])

# Define the configuration of parameters to test with the 
# Decision Tree Classifier
parameters_grid = dict(classify__criterion = ['gini', 'entropy'] , 
                  classify__min_samples_split = [2, 4, 6, 8, 10, 20],
                  classify__max_depth = [None, 5, 10, 15, 20],
                  classify__max_features = [None, 'sqrt', 'log2', 'auto'])

# Use GridSearchCV to find the optimal hyperparameters for the classifier
tree_clf = GridSearchCV(pipe_tree, param_grid = parameters_grid, scoring='f1', cv=10)
tree_clf.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
tree_clf.best_params_

# Create the classifier with the optimal hyperparameters as found by GridSearchCV
tree_clf = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classifer', DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=None, min_samples_split=20))
])

tree_clf.fit(features_train,labels_train)
tree_clf.steps[1][1].feature_importances_

# Test the classifier using tester.py
tester.dump_classifier_and_data(tree_clf, my_dataset, features_list)
tester.main()
