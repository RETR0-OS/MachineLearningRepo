import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

class KNearestNeighbors:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, k=8):
        self.k = k
        self.X = x_train
        self.Y = y_train

    @staticmethod
    def euclidean_distance(point1, point2):
        term_matrix = np.square(np.subtract(point1, point2))
        distance = np.sqrt(np.sum(term_matrix, axis=0))
        return distance

    def classify(self, test_point: pd.Series):
        self.X["distances"] = self.X.apply(lambda point2: self.euclidean_distance(point1=test_point, point2=point2), axis=1)
        result = pd.concat([self.X, self.Y], axis=1)
        result = result.sort_values(by=['distances'])
        result = result["Species"]
        classified_class = pd.DataFrame(result.iloc[:self.k]).value_counts()
        classified_class = classified_class.index[0]
        return classified_class

    def predict(self, x_test: pd.DataFrame):
        predictions = []
        for i in range(x_test.shape[0]):
            predictions.append(self.classify(pd.Series(x_test.iloc[i])))
        return pd.DataFrame(predictions, columns=["prediction"])

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        predictions = self.predict(x_test).reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        result = (predictions["prediction"] == y_test["Species"]).astype(int)
        accuracy = (np.sum(result) / result.shape[0]) * 100
        return accuracy


data = pd.read_csv('Dataset/Iris.csv').sample(frac=1)
data = data.replace({"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica": 3})
data_len = data.shape[0]

train_data = data.iloc[:int(0.8 * data_len)]
test_data = data.iloc[int(0.8 * data_len):]

model = KNearestNeighbors(train_data.drop(columns=['Species']), train_data['Species'], k=8)
accuracy = model.evaluate(test_data.drop(columns=['Species']), pd.DataFrame(test_data['Species'], columns=["Species"]))

print(accuracy)