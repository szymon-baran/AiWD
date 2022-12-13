import pandas as pd
import matplotlib.pyplot as plt 
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# _fileName = "SomervilleHappinessSurvey2015.csv"
_fileName = "winequality-red.csv"


def isAnyNull(data):
    res = data.isnull().any().any()
    if(res):
        sys.exit("Wykryto wartosc null")

def printValues(data):
    for column in data.columns:
        maxVal = data[column].max()
        minVal = data[column].min()
        stdVal = data[column].std() 
        meanVal = data[column].mean()
        medianVal = data[column].median()
        q1 = round(data[column].quantile(0.25), 3)
        q9 = round(data[column].quantile(0.75), 3)
        iqr = round(q9 - q1, 3)
        print(f'Dla kolumny {column} najwyzsza wartosc wynosi {maxVal}, a najnizsza {minVal}. \nOdchylenie standardowe wynosi {stdVal}. Srednia wartosc to {meanVal}, a mediana {medianVal}.\nKwantyl rzędu 0.1 wynosi {q1}, a rzędu 0.9 to {q9}. Rozstęp międzykwartylowy wynosi zatem {iqr}.\n')

def drawCharts(data):
    # Wykres zakresow wartosci
    data.plot.area(figsize=(15, 9), subplots=True)
    plt.show()
    
    # Wykresy pudelkowe
    data.boxplot(figsize=(15, 9))
    plt.show()

    # Histogramy atrybutow
    data.hist(figsize=(15, 9))
    plt.subplots_adjust(hspace=1) 
    plt.show()

    # Korelacje
    f = plt.figure(figsize=(16, 7))
    plt.matshow(data.corr(method='pearson'), fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, rotation=90)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns)
    plt.colorbar()
    plt.show()
    
    # Wykresy rozrzutu i regresja liniowa  
    X = data[data.columns[1]].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data[data.columns[2]].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    plt.figure(figsize=(16,7))
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[2])
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

def supportVectorMachine(data):
    classColName = ""
    if (_fileName == "SomervilleHappinessSurvey2015.csv"):
        classColName = "D"
    else:
        classColName = "quality"
    Y = data[classColName].values
    Y = Y.astype("int")
    X = data.drop(labels=[classColName], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2222, random_state=1) # 0.2222 x 0.9 = 0.2
    
    # model = svm.SVC(kernel='linear')
    model = svm.SVC(kernel='rbf')
       
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    
    print("Accuracy =", metrics.accuracy_score(Y_test, prediction))
    print("Recall =" , metrics.recall_score(Y_test, prediction, average='macro'))
   
    
    cv = KFold(n_splits=5, shuffle=True, random_state=1)  
    scores_acc = cross_val_score(model, X, Y, cv=cv)    
    scores_rec = cross_val_score(model, X, Y, cv=cv, scoring='recall_macro')
                                 
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_acc.mean(), scores_acc.std() * 2))
    print("Recall: %0.2f (+/- %0.2f)" % (scores_rec.mean(), scores_rec.std() * 2))

    # print(metrics.classification_report(Y_test, prediction)) 

data = pd.read_csv(_fileName, delimiter=',', header=0, encoding='utf-8', engine='python')
#isAnyNull(data)
#printValues(data)
#drawCharts(data)
supportVectorMachine(data)