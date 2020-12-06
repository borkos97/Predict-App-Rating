import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

np.random.seed(0)

classifiers = {
    'KNN': KNeighborsClassifier(15),
    'SVC': SVC(kernel='rbf'),
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=300)
}

df = pd.read_csv('scrapper/data.csv').drop('id', axis=1)


def plot_rating_graph(data):
    data['srednia_ocena'].describe()
    rcParams['figure.figsize'] = 11.7, 8.27
    g = sns.kdeplot(data['srednia_ocena'], color="Red", shade=True)
    g.set_xlabel("Średnia Ocena", size=15)
    g.set_ylabel("Częstotliwość", size=15)
    plt.title('Rozkład Średniej Oceny', size=20)
    plt.show()


def plot_rating_summary_length_graph(data):
    plt.figure(figsize=(10, 10))
    sns.regplot(x="srednia_ocena", y="dlugosc_opisu", scatter_kws={"color": "grey"},
                line_kws={"color": "purple"}, data=data)
    plt.title('Średnia Ocena vs Długość opisu', size=20)
    plt.show()


def plot_categories_graph(data):
    g = sns.countplot(x="kategoria", data=data, palette="Set1")
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
    g.set_xlabel("Kategoria", size=15)
    g.set_ylabel("Ilość", size=15)
    plt.title('Ilość aplikacji w każdej kategorii', size=20)
    plt.show()


def price_to_text(prices):
    if prices > 0.0:
        return 'Paid'
    else:
        return 'Free'


def plot_prices(data):
    prices = data['cena'].map(price_to_text).value_counts(sort=True)
    labels = prices.index
    colors = ["yellow", "red"]
    explode = (0.1, 0.0)
    rcParams['figure.figsize'] = 8, 8
    rcParams.update({'font.size': 15})
    plt.pie(prices, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=270, )
    plt.title('Procent darmowych i płatnych aplikacji w Google Store', size=18)
    plt.show()


def plot_supported_android(data):
    g = sns.countplot(x="wspierany_android", data=data, palette="Set1")
    g.set_xticklabels(g.get_xticklabels(), rotation=0, ha="right")
    plt.title('Ilość aplikacji pod daną wersję Androida', size=20)
    plt.show()


def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x) * 1024
        return x
    elif 'k' in size:
        x = size[:-1]
        x = float(x)
        return x
    else:
        return None


def change_supported_version(version):
    if version == 'VARY':
        return None
    if '.' in version:
        return version.split('.')[0]
    return None


def plot_confusion_matrix(cm):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=range(1, 6),
           yticklabels=range(1, 6),
           xlabel='True label',
           ylabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    plt.show()


def plot_feature_importance(data, cls):
    top_n = 15
    importances = cls.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_names = []
    for f in range(X.shape[1]):
        features_names.append(data.columns[indices[f]])
    plt.figure()
    plt.title("Feature importances")
    plt.bar(features_names[:top_n], importances[indices][:top_n],
            color="r", align="center")
    plt.xticks(range(top_n), features_names[:top_n], rotation=90, size=12)
    plt.show()


df['rozmiar_aplikacji'] = df['rozmiar_aplikacji'].map(change_size)
df['wspierany_android'] = df['wspierany_android'].map(change_supported_version)

df.dropna(inplace=True)

df = df.groupby('kategoria').filter(lambda x: len(x) > 300)
df = df.groupby('grupa_docelowa').filter(lambda x: len(x) > 50)
df = df.groupby('wspierany_android').filter(lambda x: len(x) > 50)

plot_categories_graph(df)
plot_rating_graph(df)
plot_rating_summary_length_graph(df)
plot_prices(df)
plot_supported_android(df)

df = pd.get_dummies(df, columns=['kategoria', 'grupa_docelowa'])
scaler = MinMaxScaler()

y = df['srednia_ocena'].astype('int')
X = df.drop('srednia_ocena', axis=1)
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def plot_scores(scores, labels):
    plt.ylabel('Accuracy')
    bars = plt.bar(labels, scores, align="center")
    for bar in bars:
        yval = round(bar.get_height(), 2)
        plt.text(bar.get_x(), yval + .005, yval)
    plt.xticks(range(len(scores)), labels, rotation=90, size=12)
    plt.show()


scores = []
for name, classifier in classifiers.items():
    cls = classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    scores.append(score)
    cm = confusion_matrix(y_test, classifier.predict(X_test))

plot_scores(scores, classifiers.keys())
plot_confusion_matrix(cm)
plot_feature_importance(df, cls)
