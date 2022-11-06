import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(gamma='auto'))
class SVM():
    def __init__(self):
        self.model=sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(gamma='auto'))
    def fit(self,xtrain,ytrain):
        self.model.fit(xtrain, ytrain)
    def __call__(self,xtest):
        return self.model.predict(xtest)