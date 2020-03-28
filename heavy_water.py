import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
#preprocessing steps
from sklearn import tree
with open('shuffled-full-set-hashed.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
truth_labels=[]
corpus=[]
values=[]

for line in data:
    truth_labels.append(line[0])
    corpus.append(line[1])
    value=line[1].split(' ')
    values.append(value)
#tf_id vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
le = preprocessing.LabelEncoder()
le.fit(truth_labels)
y=le.fit_transform(truth_labels)
#LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#clf = svm.SVC()
#clf.fit(X_train,y_train)
clf = LogisticRegression(random_state=0).fit(X_train,y_train)
clf.fit(X_train,y_train)
#testing
prediction=clf.predict(X_test)
