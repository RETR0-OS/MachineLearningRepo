import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

class LinearRegression:
    def __init__(self, x_train, y_train, epochs=20, alpha=0.01):
        self.x_train = pd.DataFrame(x_train)
        self.y_train = pd.DataFrame(y_train).values.reshape(-1, 1)
        self.shape = x_train.shape
        self.x_scale_factor = self.x_train.max(axis=0)
        self.y_scale_factor = self.y_train.max()
        self.x_train /= self.x_scale_factor
        self.y_train /= self.y_scale_factor
        self.weight_matrix = np.random.rand(1, self.shape[1])
        self.bias_matrix = np.random.rand(1, 1)
        self.epochs = epochs
        self.alpha = alpha
        self.error = 0

    def train(self):
        for i in range(self.epochs):
            predictions = np.dot(self.x_train, self.weight_matrix.T) + self.bias_matrix
            error_matrix = ((self.y_train - predictions) ** 2) / self.shape[0]
            self.error = np.sum(error_matrix)
            weight_gradient = -2 * np.dot((self.y_train - predictions).T, self.x_train) / self.shape[0]
            bias_gradient = -2 * np.mean(self.y_train - predictions)
            self.weight_matrix -= self.alpha * weight_gradient
            self.bias_matrix -= self.alpha * bias_gradient

            if i % 10 == 0:
                print(f"Epoch {i} \t error: {self.error}")
        print(self.weight_matrix)
        print(self.bias_matrix)

    def predict(self, x_features):
        x_features = pd.DataFrame(x_features)
        x_features /= self.x_scale_factor
        y_predictions = np.dot(x_features, self.weight_matrix.T) + self.bias_matrix
        y_predictions *= self.y_scale_factor
        return y_predictions

    def evaluate(self, x_test, y_test):
        x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test).values.reshape(-1, 1)
        x_test /= self.x_scale_factor
        y_predict = self.predict(x_test)
        rmse = np.sqrt(np.mean((y_predict - y_test) ** 2))
        return rmse

# Example usage
train_data = pd.read_csv('Dataset/House price/df_train.csv')
train_data = train_data.drop('date', axis=1)
x_train = train_data.drop(columns=['price']).sample(frac=1)
x_train = x_train.replace({True: 1, False: 0}).astype(int)
y_train = train_data['price'].sample(frac=1)
model = LinearRegression(x_train, y_train, epochs=500, alpha=0.1)
model.train()

test_data = pd.read_csv('Dataset/House price/df_test.csv')
x_test = test_data.drop(columns=['price', 'date'])
x_test = x_test.replace({True: 1, False: 0}).astype(int)
y_test = test_data['price']

print(f"Root Mean Squared Error: {model.evaluate(x_test, y_test)}")