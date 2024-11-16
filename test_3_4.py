import os
import pandas
import seaborn
import matplotlib.pyplot as pyplot
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
# input_data = pandas.read_csv(r'Provided_Data\opel_corsa_clean_combined.csv')
input_data = pandas.read_csv(r'Provided_Data\peugeot_207_clean_combined.csv')

# Output CSV file path
# report_file_path = r'results\Test_3\Test_3.csv'
report_file_path = r'results\Test_4\Test_4.csv'

# Split features from labels (column heading 'roadSurface')
features_matrix = input_data.drop(columns=['roadSurface'])
labels_matrix = input_data['roadSurface']

# Split the data into training and testing matrixes
f_m_train, f_m_test, l_m_train, l_m_test = train_test_split(features_matrix, labels_matrix, test_size=0.3, random_state=1)

# Scale the features, 0.0 - 1.0
scaler = StandardScaler()
f_m_train = scaler.fit_transform(f_m_train)
f_m_test = scaler.transform(f_m_test)

# Initialise scikit-learn's classifier methods
s_v_m_classifier_l = SVC(kernel = 'linear', random_state=1)
s_v_m_classifier_p = SVC(kernel = 'poly', random_state=1)
s_v_m_classifier_r = SVC(kernel = 'rbf', random_state=1)
s_v_m_classifier_s = SVC(kernel = 'sigmoid', random_state=1)

# Train classifiers
s_v_m_classifier_l.fit(f_m_train, l_m_train)
s_v_m_classifier_p.fit(f_m_train, l_m_train)
s_v_m_classifier_r.fit(f_m_train, l_m_train)
s_v_m_classifier_s.fit(f_m_train, l_m_train)

# Generate predictions
s_v_m_prediction_l = s_v_m_classifier_l.predict(f_m_test)
s_v_m_prediction_p = s_v_m_classifier_p.predict(f_m_test)
s_v_m_prediction_r = s_v_m_classifier_r.predict(f_m_test)
s_v_m_prediction_s = s_v_m_classifier_s.predict(f_m_test)

# Generate classification report
# Report 1
s_v_m_report_l = classification_report(l_m_test, s_v_m_prediction_l, output_dict=True)
report_dataframe_1 = pandas.DataFrame(s_v_m_report_l).transpose()
# report_dataframe_1['Test_Name'] = 'Test_3_Linear'
report_dataframe_1['Test_Name'] = 'Test_4_Linear'

# Report 2
s_v_m_report_p = classification_report(l_m_test, s_v_m_prediction_p, output_dict=True)
report_dataframe_2 = pandas.DataFrame(s_v_m_report_p).transpose()
# report_dataframe_2['Test_Name'] = 'Test_3_Polynomial'
report_dataframe_2['Test_Name'] = 'Test_4_Polynomial'

# Report 3
s_v_m_report_r = classification_report(l_m_test, s_v_m_prediction_r, output_dict=True)
report_dataframe_3 = pandas.DataFrame(s_v_m_report_r).transpose()
# report_dataframe_3['Test_Name'] = 'Test_3_RBF'
report_dataframe_3['Test_Name'] = 'Test_4_RBF'

# Report 4
s_v_m_report_s = classification_report(l_m_test, s_v_m_prediction_s, output_dict=True)
report_dataframe_4 = pandas.DataFrame(s_v_m_report_s).transpose()
# report_dataframe_4['Test_Name'] = 'Test_3_Sigmoid'
report_dataframe_4['Test_Name'] = 'Test_4_Sigmoid'

# Combine reports
combine_report_dataframe = pandas.concat([report_dataframe_1, report_dataframe_2, report_dataframe_3, report_dataframe_4])

# Save combined Report
os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
combine_report_dataframe.to_csv(report_file_path)

# Generate confussion matrixes
s_v_m_c_m_l = confusion_matrix(l_m_test, s_v_m_prediction_l)
s_v_m_c_m_p = confusion_matrix(l_m_test, s_v_m_prediction_p)
s_v_m_c_m_r = confusion_matrix(l_m_test, s_v_m_prediction_r)
s_v_m_c_m_s = confusion_matrix(l_m_test, s_v_m_prediction_s)

# Plot Confusion Matrixes
pyplot.figure(figsize=(12, 7))

# Plot 1
pyplot.subplot(2, 2, 1)
seaborn.heatmap(s_v_m_c_m_l, annot=True, fmt='d', cmap='Blues')
pyplot.title('SVM Classifier: Linear Kernel')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

# Plot 2
pyplot.subplot(2, 2, 2)
seaborn.heatmap(s_v_m_c_m_p, annot=True, fmt='d', cmap='Blues')
pyplot.title('SVM Classifier: Polynomial Kernel')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

# Plot 3
pyplot.subplot(2, 2, 3)
seaborn.heatmap(s_v_m_c_m_r, annot=True, fmt='d', cmap='Blues')
pyplot.title('SVM Classifier: RBF Kernel')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

# Plot 3
pyplot.subplot(2, 2, 4)
seaborn.heatmap(s_v_m_c_m_s, annot=True, fmt='d', cmap='Blues')
pyplot.title('SVM Classifier: Sigmoid Kernel')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

pyplot.tight_layout()
pyplot.show()
