import Lib.PreprocessamentoDados as prepro
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

training_sets = "C:\\Users\Rodrigo\Documents\Projetos\Twitter Python\data\TrainingDatasets\\"
testing_sets = "C:\\Users\Rodrigo\Documents\Projetos\Twitter Python\data\TestDatasets\\"

def run_model(x,y):
    ##model = MultinomialNB()
    model = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)
    return model.fit(x,y)

if __name__ == '__main__':
    x,y = prepro.pre_processamento(training_sets+"Train3Classes.csv")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33)
    model = run_model(x_train,y_train)
    y_pred = model.predict(x_test)

    ## Confusion Matrix
    matrix = metrics.confusion_matrix(y_test,y_pred)
    print(matrix)




