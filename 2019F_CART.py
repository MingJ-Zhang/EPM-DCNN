import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef


def cart():
    X = np.load('./data/train_data.npy')
    y = np.load('./data/train_label.npy')
    X = np.reshape(X, [X.shape[0], X.shape[1] * X.shape[2]])
    y = np.reshape(y, [y.shape[0], ])

    print(X.shape)
    print(y.shape)
    Xtest = np.load('./data/test_data.npy')
    Xtest = np.reshape(Xtest, [Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2]])
    Ytest = np.load('./data/test_label.npy')
    Ytest = np.reshape(Ytest, [Ytest.shape[0], ])

    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random")
    clf = clf.fit(X, y)
    result = clf.predict(Xtest)
    result1 = clf.predict(X)
    print('train acc and recall')
    print(accuracy_score(y, result1))
    print(recall_score(y, result1, average='weighted'))

    print('test acc and recall')
    print(accuracy_score(Ytest, result))
    print(recall_score(Ytest, result, average='weighted'))
    print(precision_score(Ytest, result, average='weighted'))
    print(f1_score(Ytest, result, average='weighted'))
    print("MCC=======================================")
    print(matthews_corrcoef(Ytest, result)) 

if __name__ == '__main__':
    cart()
