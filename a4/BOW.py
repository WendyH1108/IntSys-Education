from pre_process import sen_tokenizer, predictors, read_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



if __name__ == "__main__":
    tfidf_vector = TfidfVectorizer(tokenizer = sen_tokenizer)
    tfidf_trans = TfidfTransformer()
    classifier = LogisticRegression(multi_class='multinomial', max_iter=300,  random_state=1)
    # classifier = RandomForestClassifier(n_estimators = 50, min_samples_leaf=2)
    # classifier = SVC(decision_function_shape='ovr', C=0.01, kernel = 'linear', verbose=True)
    #to better re-organize and understan sinppets in sentence 
    bow_vector = CountVectorizer(tokenizer = sen_tokenizer, ngram_range=(1,2))
    pipe = Pipeline([("preprocesser", predictors()),
                 ("vectorizer", tfidf_vector),
                 ('normalizer', tfidf_trans),
                 ('classifier', classifier)])
    X_train, y_train = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/train.csv')
    X_val, y_val = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/val.csv')
    pipe.fit(X_train[:15000],y_train[:15000])
    origin = pipe.predict(X_train[:15000])
    predict = pipe.predict(X_val[:15000])
    print("Logistic Regression Accuracy Train:",metrics.accuracy_score(y_train[:15000], origin))
    print("Logistic Regression Accuracy Val:",metrics.accuracy_score(y_val[:15000], predict))
