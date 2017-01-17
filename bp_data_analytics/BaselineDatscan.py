from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, cross_validation
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt


def main():
    # Set seed
    # np.random.seed(0)

    # Create the training & test sets from files
    train = pd.read_csv("data/all_baseline.csv")
    
    train_on = train[train["ONOFF_V04"] == 1]
    train_off = train[train["ONOFF_V04"] == 0]

    # Preliminary data diagnostics
    print("PRELIMINARY DATA DIAGNOSTICS:")
    
    # On state
    print("\nOn state:\n")
    print("Info:")
    print(train_on.info())
    print("\nDescribe:")
    print(train_on.describe())
    print("\nValue counts NP3BRADY_V04:")
    print(pd.value_counts(train_on["NP3BRADY_V04"]))
    
    # Off state
    print("\nOff state:\n")
    print("Info:")
    print(train_off.info())
    print("\nDescribe:")
    print(train_off.describe())
    print("\nValue counts NP3BRADY_V04:")
    print(pd.value_counts(train_off["NP3BRADY_V04"]))
    print("")

    # Generate new features
    # new_features(train_on)

    # The columns we'll use to predict the target
    predictors = ["PATNO", "CAUDATE_R_SC", "NP3BRADY_SC", "CAUDATE_L_SC", "PUTAMEN_R_SC", "PUTAMEN_L_SC", "ONOFF_V04"]

    def scale(data, features):
        data.loc[:, features] = MinMaxScaler(copy=False).fit_transform(data[features])

    # On/off state
    print("On/off state:\n")

    # Scale the data
    # scale(train, ["CAUDATE_R_SC", "CAUDATE_L_SC", "PUTAMEN_R_SC", "PUTAMEN_L_SC"])

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["NP3BRADY_V04"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=450, min_samples_split=25, min_samples_leaf=2, oob_score=True)

    # Fit the algorithm to the data
    rf.fit(train_predictors, train_target)

    # Perform feature selection
    selector = SelectKBest(f_classif, k='all')
    selector.fit(train_predictors, train_target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    for feature, imp in zip(predictors, scores):
        print(feature, imp)

    # Metrics:
    print("\nRANDOM FOREST METRICS:")

    # Feature importances
    print("\nFeature importances:")
    for feature, imp in zip(predictors, rf.feature_importances_):
        print(feature, imp)

    # Recursive feature elimination
    # print("\nRecursive feature elimination:")
    # rfe = RFE(rf, 5)
    # rfe = rfe.fit(train_predictors, train_target)
    # print(rfe.support_)
    # print(rfe.ranking_)

    # Base estimate
    print("\nBase score: ")
    print(rf.score(train_predictors, train_target))

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Ensemble metrics
    ensemble(rf, train_predictors, train_target, predictors)

    # # On state
    # print("On state:\n")
    #
    # # Scale the data
    # # scale(train_on, ["CAUDATE_R_SC", "CAUDATE_L_SC", "PUTAMEN_R_SC", "PUTAMEN_L_SC"])
    #
    # # Prepare predictors
    # train_on_predictors = train_on[predictors]
    #
    # # Prepare target
    # train_on_target = train_on["NP3BRADY_V04"]
    #
    # # Create and train_on the random forest
    # # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    # rf = RandomForestClassifier(n_estimators=450, min_samples_split=25, min_samples_leaf=2, oob_score=True)
    #
    # # Fit the algorithm to the data
    # rf.fit(train_on_predictors, train_on_target)
    #
    # # Perform feature selection
    # selector = SelectKBest(f_classif, k='all')
    # selector.fit(train_on_predictors, train_on_target)
    #
    # # Get the raw p-values for each feature, and transform from p-values into scores
    # scores = -np.log10(selector.pvalues_)
    # print("Univariate feature selection:")
    # for feature, imp in zip(predictors, scores):
    #     print(feature, imp)
    #
    # # Metrics:
    # print("\nRANDOM FOREST METRICS:")
    #
    # # Feature importances
    # print("\nFeature importances:")
    # for feature, imp in zip(predictors, rf.feature_importances_):
    #     print(feature, imp)
    #
    # # Recursive feature elimination
    # # print("\nRecursive feature elimination:")
    # # rfe = RFE(rf, 5)
    # # rfe = rfe.fit(train_on_predictors, train_on_target)
    # # print(rfe.support_)
    # # print(rfe.ranking_)
    #
    # # Base estimate
    # print("\nBase score: ")
    # print(rf.score(train_on_predictors, train_on_target))
    #
    # # Out of bag estimate
    # print("OOB score: ")
    # print(rf.oob_score_)
    #
    # # Ensemble metrics
    # ensemble(rf, train_on_predictors, train_on_target, predictors)
    #
    # # Off state
    # print("\nOff state:\n")
    #
    # # Scale the data
    # # scale(train_off, ["CAUDATE_R_SC", "CAUDATE_L_SC", "PUTAMEN_R_SC", "PUTAMEN_L_SC"])
    #
    # # Prepare predictors
    # train_off_predictors = train_off[predictors]
    #
    # # Prepare target
    # train_off_target = train_off["NP3BRADY_V04"]
    #
    # # Create and train_off the random forest
    # # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    # rf = RandomForestClassifier(n_estimators=450, min_samples_split=50, min_samples_leaf=2, oob_score=True)
    #
    # # Fit the algorithm to the data
    # rf.fit(train_off_predictors, train_off_target)
    #
    # # Perform feature selection
    # selector = SelectKBest(f_classif, k='all')
    # selector.fit(train_off_predictors, train_off_target)
    #
    # # Get the raw p-values for each feature, and transform from p-values into scores
    # scores = -np.log10(selector.pvalues_)
    # print("Univariate feature selection:")
    # for feature, imp in zip(predictors, scores):
    #     print(feature, imp)
    #
    # # Metrics:
    # print("\nRANDOM FOREST METRICS:")
    #
    # # Feature importances
    # print("\nFeature importances:")
    # for feature, imp in zip(predictors, rf.feature_importances_):
    #     print(feature, imp)
    #
    # # Recursive feature elimination
    # # print("\nRecursive feature elimination:")
    # # rfe = RFE(rf, 5)
    # # rfe = rfe.fit(train_off_predictors, train_off_target)
    # # print(rfe.support_)
    # # print(rfe.ranking_)
    #
    # # Base estimate
    # print("\nBase score: ")
    # print(rf.score(train_off_predictors, train_off_target))
    #
    # # Out of bag estimate
    # print("OOB score: ")
    # print(rf.oob_score_)
    #
    # # Ensemble metrics
    # ensemble(rf, train_off_predictors, train_off_target, predictors)


def ensemble(rf, X, y, predictors):
    # Classifiers
    lr = LogisticRegression()
    svm = SVC(probability=True)
    gnb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=7)
    gb = GradientBoostingClassifier(n_estimators=25, max_depth=2)

    # Ensemble
    eclf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn), ('gb', gb)], voting='soft')
    # eclf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')

    print("\nENSEMBLE METRICS:")

    # Feature importances
    print("\nFeature importances:")
    for feature, imp in zip(predictors, rf.feature_importances_):
        print(feature, imp)

    # Cross validation accuracies
    print("\nCross validation:\n")
    for clf, label in zip([rf, lr, svm, gnb, knn, gb, eclf], ['Random Forest', 'Logistic Regression', 'SVM', 'naive Bayes', 'kNN', 'Gradient Boosting', 'Ensemble of RF, LR, and SVM']):
        scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20)
    print("\n80/20 split: \n")
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

    # Plot normalized confusion matrix
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    # plt.show()


if __name__ == "__main__":
    main()
