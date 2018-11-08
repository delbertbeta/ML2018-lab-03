from PIL import Image
import numpy as np
import feature
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os

if __name__ == "__main__":
    # write your code here
    features = []
    y = np.ones((500, 1))
    y = np.append(y, np.ones((500, 1)) * -1).reshape((-1, 1))
    if os.path.exists('features.dump'):
        features = AdaBoostClassifier.load('features.dump')
    else:
        pathes = np.array(list(map(lambda s: 'datasets/original/face/' + s, os.listdir('datasets/original/face'))))
        pathes = np.append(pathes, np.array(
            list(map(lambda s: 'datasets/original/nonface/' + s, os.listdir('datasets/original/nonface')))))
        for index, path in enumerate(pathes):
            with Image.open(path) as image:
                print(index, path)
                image = image.convert('L')
                image = image.resize((24, 24))
                imageData = np.array(image)
                npd = feature.NPDFeature(imageData)
                features.append(npd.extract())
        AdaBoostClassifier.save(features, 'features.dump')

    features = np.array(features)
    print(features.shape)

    X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.25)

    classifier = AdaBoostClassifier(DecisionTreeClassifier, 5)
    classifier.fit(X_train, y_train)

    score = classifier.predict_scores(X_val, y_val)
    predict = classifier.predict(X_val)

    y_val = np.array(list(map(lambda x: int(x), y_val.reshape(1, -1)[0])))
    predict = np.array(list(map(lambda x: int(x), predict.reshape(1, -1)[0])))

    print(predict)
    print(y_val)

    reportContent = 'score = ' + str(score) + '\n'
    reportContent += classification_report(y_val, predict)

    with open('classifier_report.txt', 'w') as report:
        report.write(reportContent)

    pass

