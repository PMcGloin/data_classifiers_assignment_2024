import os
import pandas
import seaborn
import matplotlib.pyplot as pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
# input_data = pandas.read_csv(r'Provided_Data\opel_corsa_clean_combined.csv')
input_data = pandas.read_csv(r'Provided_Data\peugeot_207_clean_combined.csv')

# Output CSV file path
# report_file_path = r'results\Test_1\Test_1.csv'
report_file_path = r'results\Test_2\Test_2.csv'

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
d_t_classifier_g = DecisionTreeClassifier(criterion='gini', random_state=1)
d_t_classifier_e = DecisionTreeClassifier(criterion='entropy', random_state=1)
d_t_classifier_l = DecisionTreeClassifier(criterion='log_loss', random_state=1)

# Train classifiers
d_t_classifier_g.fit(f_m_train, l_m_train)
d_t_classifier_e.fit(f_m_train, l_m_train)
d_t_classifier_l.fit(f_m_train, l_m_train)

# Generate predictions
d_t_prediction_g = d_t_classifier_g.predict(f_m_test)
d_t_prediction_e = d_t_classifier_e.predict(f_m_test)
d_t_prediction_l = d_t_classifier_l.predict(f_m_test)

# Generate classification report
# Report 1
d_t_report_g = classification_report(l_m_test, d_t_prediction_g, output_dict=True)
report_dataframe_1 = pandas.DataFrame(d_t_report_g).transpose()
# report_dataframe_1['Test_Name'] = 'Test_1_Gini'
report_dataframe_1['Test_Name'] = 'Test_2_Gini'

# Report 2
d_t_report_e = classification_report(l_m_test, d_t_prediction_e, output_dict=True)
report_dataframe_2 = pandas.DataFrame(d_t_report_e).transpose()
# report_dataframe_2['Test_Name'] = 'Test_1_Entropy'
report_dataframe_2['Test_Name'] = 'Test_2_Entropy'

# Report 3
d_t_report_l = classification_report(l_m_test, d_t_prediction_l, output_dict=True)
report_dataframe_3 = pandas.DataFrame(d_t_report_l).transpose()
# report_dataframe_3['Test_Name'] = 'Test_1_Log'
report_dataframe_3['Test_Name'] = 'Test_2_Log'

# Combine reports
combine_report_dataframe = pandas.concat([report_dataframe_1, report_dataframe_2, report_dataframe_3])

# Save combined Report
os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
combine_report_dataframe.to_csv(report_file_path)

# Generate confussion matrixes
d_t_c_m_g = confusion_matrix(l_m_test, d_t_prediction_g)
d_t_c_m_e = confusion_matrix(l_m_test, d_t_prediction_e)
d_t_c_m_l = confusion_matrix(l_m_test, d_t_prediction_l)

# Plot Confusion Matrixes
pyplot.figure(figsize=(12, 7))

# Plot 1
pyplot.subplot(2, 2, 1)
seaborn.heatmap(d_t_c_m_g, annot=True, fmt='d', cmap='Blues')
pyplot.title('Decision Tree Classifier: Gini Index')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

# Plot 2
pyplot.subplot(2, 2, 2)
seaborn.heatmap(d_t_c_m_e, annot=True, fmt='d', cmap='Blues')
pyplot.title('Decision Tree Classifier: Entropy')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

# Plot 3
pyplot.subplot(2, 2, 3)
seaborn.heatmap(d_t_c_m_l, annot=True, fmt='d', cmap='Blues')
pyplot.title('Decision Tree Classifier: Log Loss')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')
pyplot.tight_layout()
pyplot.show()
