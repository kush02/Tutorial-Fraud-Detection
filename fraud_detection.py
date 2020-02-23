import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, plot_precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from imblearn.over_sampling import SMOTE
import itertools
from collections import Counter



def describe_fields(data,field_type='numeric'):
	"""
		input: Pandas DataFrame, field_type = ['numeric', 'categorical', 'boolean']
		output: Pandas DataFrame
	"""

	# calculate and print descriptive statistics for numerical fields
	if field_type =='numeric':
		fields = data.describe(include=[np.number])
		print('Describing numerical fields:')
		print(fields)
	
	# calculate and print descriptive statistics for categorical fields
	elif field_type =='categorical': 
		fields = data.describe(include=['O'])
		print('Describing categorical fields:')
		print(fields)
	
	# calculate and print descriptive statistics for boolean fields
	elif field_type =='boolean':
		fields = data.describe(include=[bool])
		print('Describing boolean fields:')
		print(fields)
	
	else:
		print("Please specify field type")
	
	return


def col_count_plot(data,col_name='',title=''):
	"""
		input: Pandas DataFrame, strings
		output: bar plot
	"""

	# check if column name is provided
	if not col_name:
 		print("Enter column name")
 		return

	# bar plot of the column
	sns.countplot(col_name,data=data)

	plt.title(title)

	# show plot
	plt.show()

	return


def inspect_every_col_against_bin_target_var(data,target_var=""):
	"""
		input: Pandas DataFrame, string
		output: boxplot and distribtion plot
	"""

	# check if target variable is provided
	if not target_var:
		print('Enter target variable name')
		return

	# plot a boxplot and distribution plot for each feature against the target variable
	for i in data.columns:
		feature_name = str(i)
		
		f,(ax1,ax2) = plt.subplots(1,2)
		
		sns.boxplot(x=target_var,y=feature_name,data=data,ax=ax1)
		ax1.set_title(feature_name + ' ' + 'boxplot')
		
		sns.distplot(data[data[target_var]==0][feature_name],ax=ax2,label='False')
		sns.distplot(data[data[target_var]==1][feature_name],ax=ax2,label='True')
		ax2.set_title(feature_name + ' ' + 'distribution')
		plt.ylabel('Density')
		plt.legend()
		
		plt.show()
	
	return


def visualize_corr_matrix(corr_matrix):
	"""
		input: numpy ndarray
		output: heatmap plot
	"""
	
	sns.heatmap(corr_matrix, cmap='Blues', annot=True)	

	plt.show()

	return


def grid_search(model,parameters,X_train,y_train,metric='accuracy',cv=2,verbose=1):
	"""
		input: model = SKLearn estimator object, parameters = dict,X_train = numpy ndarray, y_train = numpy array, metric = string, cv = int, verbose = int
		output: SKLearn estimator object
	"""

	# perform grid search
	cv = GridSearchCV(model,parameters,scoring=metric,cv=cv,verbose=verbose)
	cv.fit(X_train,y_train)

	# get best estimator from the grid search
	best_model = cv.best_estimator_

	return best_model


def print_classifier_metrics(y_test,y_pred,name="",average='binary'):
	"""
		input: y_test = numpy array, y_pred = numpy array, average = list of metrics from SKLearn's documentation
		output: accuracy = float, recall = float, precision = float, f1 = float
	"""
	print("Accuracy score for %s: %f" %(name,accuracy_score(y_test,y_pred)))
	print("Recall score for %s: %f" % (name,recall_score(y_test,y_pred,average=average)))
	print("Precision score for %s: %f" % (name,precision_score(y_test,y_pred,average=average)))
	print("F-1 score for %s: %f" % (name,f1_score(y_test,y_pred,average=average)))

	return


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
		input: cm = numpy ndarray, classes = list of classes, normalize = boolean,
		       title = title of plot, cmap = color coding for the plot
		output: Confusion Matrix image
	"""

	# display magnitude or fraction
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	# creating plot
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	# displaying plot
	plt.show()

	return


def plot_roc_curve(y_test,decision_function,name=""):
	"""
		input: y_test = numpy array, decision_function = numpy ndarray, name = name of the classifier
		output: ROC curve plot
	"""
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	# calculate roc values
	fpr, tpr, thresholds = roc_curve(y_test, decision_function)
	roc_auc = auc(fpr, tpr)

	# create roc cruve
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
	plt.title('%s ROC curve' % name);plt.legend(loc="lower right")

	# display roc curve
	plt.show()

	return


def sklearn_precision_recall_curve(model, X_test, y_test):
	"""
		input: model = SKLearn estimator object, X_test = numpy nd.array, y_test = numpy array
		output: plot
	"""

	# plot the precision-recall curve
	plot_precision_recall_curve(model, X_test, y_test)
	
	plt.show()

	return


def plot_feature_importances(feature_names,importance,name=""):
	"""
		input: feature = list of column names, importance = numpy array, name = name of classifier
		output: bar plot
	"""

	# put features and importance values in a DataFrame.
	feature_importances = pd.DataFrame({'feature':feature_names, 'importance':importance.flatten()})

	# sort DataFrame in descending order to see the most influential features
	feature_importances.sort_values(by='importance', ascending=False, inplace=True)
	feature_importances.set_index('feature', inplace=True, drop=True)
	
	# plot the importances
	feature_importances.plot(kind='bar')
	plt.title(name + ' ' + 'feature importances')
	plt.show()    

	return



def main():
	# Set random seed
	np.random.seed(42)

	# Load data
	credit_card = pd.read_csv('creditcard.csv')

	# Print column names
	print("Column names: ", credit_card.columns)

	# Look at data types of each column
	print("Inspecting columns")
	print(credit_card.info())

	# Change data type of target variable to boolean
	credit_card = credit_card.astype({'Class':bool})

	# Count the number of duplicated transactions
	print("Number of duplicate transactions: ",credit_card.duplicated().sum())

	# Remove duplicate transactions
	credit_card = credit_card.drop_duplicates()

	# Calculate descriptive statistics for Amount and Time
	describe_fields(credit_card[['Time','Amount']])

	# Calculate descriptive statistics for Class
	describe_fields(credit_card['Class'],field_type='categorical')

	# Visualize class imbalance 
	col_count_plot(credit_card,col_name='Class',title='Class Distribution')

	# Visualize distribution of each column and correlation with the target variable
	inspect_every_col_against_bin_target_var(credit_card.drop('Class',axis=1),target_var='Class')

	# Columns to remove after looking at the plots
	cols_to_drop = ['Time','V1','V5','V6','V8','V13','V15','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

	# Remove columns
	credit_card = credit_card.drop(cols_to_drop,axis=1)

	# Visualize correlation matrix to check for multicollinearity (highly correlated predictor variables)
	# After looking at correlation matrix, decided to not remove any more features since features are mostly independent
	corr_matrix = credit_card.corr()
	visualize_corr_matrix(corr_matrix)

	# Shuffle data to possibly improve cross validation accuracy
	credit_card = shuffle(credit_card).reset_index().drop('index',axis=1)

	# Split dataset into predictors and target variables
	X = credit_card.drop('Class',axis=1)
	y = credit_card['Class']
	
	# Split data into training and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Oversample the training data
	X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train,y_train)
	
	# Define Logistic Regression model and parameters for grid search
	lr = LogisticRegression(random_state=42)
	parameters = {'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10],'solver':['liblinear','lbfgs']}

	# Find best Logistic Regression model using grid search with cross validation
	best_model_lr = grid_search(lr,parameters,X_train_res,y_train_res,metric='recall',cv=5,verbose=5)
	print("Best Logistic Regression model: ", best_model_lr)

	# Train best model on whole training set and make predictions on test set
	y_pred = best_model_lr.fit(X_train_res,y_train_res).predict(X_test)
	
	# Assess the quality of the predictions
	cm = confusion_matrix(y_test,y_pred)
	classes = ['Not Fraud','Fraud']
	print_classifier_metrics(y_test,y_pred,name='Logistic Regression')
	plot_confusion_matrix(cm,classes=classes,title='Logistic Regression Confusion Matrix')
	plot_roc_curve(y_test,best_model_lr.decision_function(X_test),name="Logistic Regression") 
	sklearn_precision_recall_curve(best_model_lr, X_test, y_test)

	# Most informative, discriminative features for Logistic Regression model
	plot_feature_importances(X.columns,best_model_lr.coef_,name='Logistic Regression')
	
	# Define Random Forest model and parameters for grid search
	rf = RandomForestClassifier(random_state=42,oob_score=True)
	parameters = {'n_estimators':[10,20],'criterion':['gini','entropy'],'max_depth':[5,10],'min_samples_split':[5,10]}

	# Find best Logistic Regression model using grid search with cross validation
	best_model_rf = grid_search(rf,parameters,X_train_res,y_train_res,metric='recall',cv=5,verbose=5)
	print("Best Random Forest model: ", best_model_rf)

	# Train best model on whole training set and make predictions on test set
	y_pred = best_model_rf.fit(X_train_res,y_train_res).predict(X_test)

	# Assess the quality of the predictions
	cm = confusion_matrix(y_test,y_pred)
	print_classifier_metrics(y_test,y_pred,name='Random Forest')
	plot_confusion_matrix(cm,classes=classes,title='Random Forest Confusion Matrix')
	plot_roc_curve(y_test,best_model_rf.predict_proba(X_test)[:,1],name="Random Forest") 
	sklearn_precision_recall_curve(best_model_rf, X_test, y_test)

	# Most informative, discriminative features for Random Forest model
	plot_feature_importances(X.columns,best_model_rf.feature_importances_,name='Random Forest')
	
	# Visualize a decision tree in the random forest. Use onlineconvertfree.com to convert dot file to png
	tree = best_model_rf.estimators_[0]
	export_graphviz(tree,out_file='tree.dot',feature_names=X.columns,class_names=classes,filled=True,rounded=True)


if __name__ == '__main__':
    main()
