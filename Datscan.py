from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt


def main():
    # Set seed
    np.random.seed(0)

    # Create the training & test sets from files
    train = pd.read_csv("data/all_visits_practice.csv")

    # Preliminary data diagnostics
    print("PRELIMINARY DATA DIAGNOSTICS:\n")
    print("Info:")
    print(train.info())
    print("\nDescribe:")
    print(train.describe())
    # print("Unique EVENT_ID")
    # print(train["EVENT_ID"].unique())
    print("\nValue counts NP3BRADY:")
    print(pd.value_counts(train["NP3BRADY"]))
    print("")

    # Encode EVENT_ID to numeric
    train["EVENT_ID"] = preprocessing.LabelEncoder().fit_transform(train["EVENT_ID"])

    # Remove the class with only a single label
    train = train[train.NP3BRADY != 4]

    # Generate new features
    # new_features(train)

    # The columns we'll use to predict the target
    predictors = ["PATNO", "EVENT_ID", "CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"]

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["NP3BRADY"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=350, min_samples_split=50, min_samples_leaf=25, oob_score=True)

    # Fit the algorithm to the data
    rf.fit(train_predictors, train_target)

    # Metrics:
    print("RANDOM FOREST METRICS: \n")

    # Perform feature selection
    selector = SelectKBest(f_classif, k='all')
    selector.fit(train_predictors, train_target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    for feature, imp in zip(predictors, scores):
        print(feature, imp)

    # Feature importances
    print("\nFeature importances:")
    for feature, imp in zip(predictors, rf.feature_importances_):
        print(feature, imp)

    # Base estimate
    print("\nBase score: ")
    print(rf.score(train_predictors, train_target))

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Ensemble metrics
    print("\nENSEMBLE METRICS:\n")
    ensemble(rf, train_predictors, train_target)


def ensemble(rf, X, y):
    # Classifiers
    lr = LogisticRegression()
    svm = SVC(probability=True)
    gnb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=7)
    gb = GradientBoostingClassifier(n_estimators=350)

    # Ensemble
    # eclf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm), ('gnb', gnb), ('knn', knn), ('gb', gb)], voting='soft')
    eclf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')

    # Cross validation accuracies
    print("Cross validation:\n")
    for clf, label in zip([rf, lr, svm, gnb, knn, gb, eclf], ['Random Forest', 'Logistic Regression', 'SVM', 'naive Bayes', 'kNN', 'Gradient Boosting', 'Ensemble of RF, LR, and SVM']):
        scores = cross_validation.cross_val_score(clf, X, y, cv=2, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)
    print("\n75/25 split: \n")
    y_pred_ensemble = eclf.fit(X_train, y_train).predict(X_test)

    # Accuracies
    for clf, label in zip([rf, lr, svm, gnb, knn, gb, eclf], ['Random Forest', 'Logistic Regression', 'SVM', 'naive Bayes', 'kNN', 'Gradient Boosting', 'Ensemble of RF, LR, and SVM']):
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        print("Accuracy: %0.2f [%s]" % (accuracy_score(y_test, y_pred), label))

    # Classification report
    print("\nClassification report:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(classification_report(y_test, y_pred_ensemble))

    # Print a confusion matrix for ensemble
    print("\nConfusion matrix of ensemble (rows: true, cols: pred)")

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, [0, 1, 2, 3], rotation=45)
        plt.yticks(tick_marks, [0, 1, 2, 3])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization:')
    print(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix:')
    print(cm_normalized)
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    plt.show()


if __name__ == "__main__":
    main()
