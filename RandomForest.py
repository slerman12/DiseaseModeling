from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def main():
    #create the training & test sets
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    #prepare data
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1   
    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2

    test["Age"] = test["Age"].fillna(test["Age"].median())
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    test["Embarked"] = test["Embarked"].fillna("S")
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # The columns we'll use to predict the target
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    train_predictors = train[predictors]

    #prepare target
    train_target = train["Survived"]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_predictors, train_target)

    predictions = rf.predict(test[predictors])

    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0

    # accuracy = sum(predictions[predictions == test["Survived"]]) / len(predictions)
    # print(accuracy)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv("data/kaggle.csv", index=False)

if __name__=="__main__":
    main()
