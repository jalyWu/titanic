#coding=utf-8
__author__ = 'Administrator'

import pandas as pd
from pandas import Series,DataFrame

import numpy as np
from datetime import datetime
import time
import re
import operator
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

def harmonize_data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic

def linearRegression(titanic, predictors):
    kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
    alg = LinearRegression()
    predictions = []
    for train, test in kf:
        train_predictors = (titanic[predictors].iloc[train, :])
        train_target = titanic["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(titanic[predictors].iloc[test, :])
        predictions.append(test_predictions)

    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    accuracy = sum(predictions == titanic['Survived'])/float(len(predictions))
    return accuracy

def create_submission(labels, predictions, filename):
    submission = pd.DataFrame({
        "PassengerId": labels,
        "Survived": predictions
    })
    submission.to_csv(filename+'.csv', index=False)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

family_id_mapping = {}
def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def trainNewFeatures(titanic):
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    titles = titanic["Name"].apply(get_title)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    titanic["Title"] = titles
    family_ids = titanic.apply(get_family_id, axis=1)
    family_ids[titanic["FamilySize"] < 3] = -1
    titanic["FamilyId"] = family_ids
    return titanic

def testNewFeatures(titanic):
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    titles = titanic["Name"].apply(get_title)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    titanic["Title"] = titles
    family_ids = titanic.apply(get_family_id, axis=1)
    family_ids[titanic["FamilySize"] < 3] = -1
    titanic["FamilyId"] = family_ids
    return titanic

def selectBestFeatures(titanic):
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic[predictors], titanic["Survived"])
    scores = -np.log10(selector.pvalues_)
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

def ensembling(titanic):
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]
    kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
    predictions = []
    for train, test in kf:
        train_target = titanic["Survived"].iloc[train]
        full_test_predictions = []
        for alg, predictors in algorithms:
            alg.fit(titanic[predictors].iloc[train, :], train_target)
            test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:,1]
            full_test_predictions.append(test_predictions)
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        test_predictions[test_predictions <= .5] = 0
        test_predictions[test_predictions > .5] = 1
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    accuracy = sum(predictions == titanic["Survived"])/float(len(predictions))
    return accuracy

if __name__ == '__main__':
    # get titanic & test csv files as a DataFrame
    titanic_df = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
    test_df = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

    # The columns we'll use to predict the target
    # predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    titanic = harmonize_data(titanic_df)
    # print linearRegression(titanic, predictors) #kfold

    titanic = trainNewFeatures(titanic)
    # selectBestFeatures(titanic)
    # predictors = ["Pclass", "Sex", "Fare", "Title"]

    # print ensembling(titanic)

    #local test
    # alg = LogisticRegression(random_state=1)
    # alg = RandomForestClassifier(random_state=1, n_estimators=30, max_depth=3)
    # scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
    # print(scores.mean())
    # exit()

    test = harmonize_data(test_df)
    test = testNewFeatures(test)

    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]],
        [RandomForestClassifier(random_state=1, n_estimators=30, max_depth=3), predictors]
    ]
    full_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors], titanic["Survived"])
        predictions = alg.predict_proba(test[predictors].astype(float))[:, 1]
        full_predictions.append(predictions)
    predictions = (full_predictions[0] * 3 + full_predictions[1] + full_predictions[2]*2) / 6
    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    predictions = predictions.astype(int)

    #single alg
    # alg.fit(titanic[predictors], titanic['Survived'])
    # predictions = alg.predict(test[predictors])

    filename = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    create_submission(test['PassengerId'], predictions, filename)