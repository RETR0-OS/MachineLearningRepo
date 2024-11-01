import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

class LinearRegression:
    def __init__(self, x_train, y_train, epochs=20, alpha=0.01):
        self.x_train = pd.DataFrame(x_train)
        self.y_train = pd.DataFrame(y_train).values.reshape(-1, 1)
        self.shape = x_train.shape

        # Scaling factors
        self.x_scale_factor = self.x_train.max(axis=0)
        self.y_scale_factor = self.y_train.max()

        # Scale training data
        self.x_train /= self.x_scale_factor
        self.y_train /= self.y_scale_factor

        # Initialize weight and bias matrices
        self.weight_matrix = np.random.rand(1, self.shape[1])
        self.bias_matrix = np.random.rand(1, 1)
        self.epochs = epochs
        self.alpha = alpha
        self.error = 0

    def train(self):
        for i in range(self.epochs):
            predictions = np.dot(self.x_train, self.weight_matrix.T) + self.bias_matrix  # Predicted values

            error_matrix = ((self.y_train - predictions) ** 2) / self.shape[0]  # Find MSE
            self.error = np.sum(error_matrix)

            # Gradient Descent
            weight_gradient = -2 * np.dot((self.y_train - predictions).T, self.x_train) / self.shape[0]  # Find gradient of weights
            bias_gradient = -2 * np.mean(self.y_train - predictions)  # Find gradient of bias

            self.weight_matrix -= self.alpha * weight_gradient  # Update weights
            self.bias_matrix -= self.alpha * bias_gradient  # Update bias

            if i % 10 == 0:
                print(f"Epoch {i} \t error: {self.error}")

    def predict(self, x_features):
        x_features = pd.DataFrame(x_features)
        x_features /= self.x_scale_factor  # Scale features
        y_predictions = np.dot(x_features, self.weight_matrix.T) + self.bias_matrix
        y_predictions *= self.y_scale_factor  # Scale back predictions
        return y_predictions

    def evaluate(self, x_test, y_test):
        x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test).values.reshape(-1, 1)
        y_predict = self.predict(x_test)
        mse = np.mean((y_predict - y_test) ** 2)
        threshhold = 3000
        accuracy = (np.sum(np.abs(y_predict - y_test) < threshhold) / len(y_test)) * 100
        return accuracy

train_data = pd.read_csv("Dataset/House price/df_train.csv")
train_data = train_data.drop("date", axis=1)
x_train = train_data.drop(columns=["price"]).sample(frac=1)
x_train["has_basement"] = x_train["has_basement"].replace({True:1, False:0}).astype(int)
x_train["renovated"] = x_train["renovated"].replace({True: 1, False: 0}).astype(int)
x_train["nice_view"] = x_train["nice_view"].replace({True:1, False:0}).astype(int)
x_train["perfect_condition"] = x_train["perfect_condition"].replace({True:1, False:0}).astype(int)
x_train["has_lavatory"] = x_train["has_lavatory"].replace({True:1, False:0}).astype(int)
x_train["single_floor"] = x_train["single_floor"].replace({True:1, False:0}).astype(int)
y_train = train_data["price"].sample(frac=1)
model = LinearRegression(x_train, y_train, epochs=500, alpha=0.1)
model.train()
test_data = pd.read_csv("Dataset/House price/df_test.csv")
x_test = test_data.drop(columns=['price', 'date'])
x_test["has_basement"] = x_test["has_basement"].replace({True:1, False:0}).astype(int)
x_test["renovated"] = x_test["renovated"].replace({True: 1, False: 0}).astype(int)
x_test["nice_view"] = x_test["nice_view"].replace({True:1, False:0}).astype(int)
x_test["perfect_condition"] = x_test["perfect_condition"].replace({True:1, False:0}).astype(int)
x_test["has_lavatory"] = x_test["has_lavatory"].replace({True:1, False:0}).astype(int)
x_test["single_floor"] = x_test["single_floor"].replace({True:1, False:0}).astype(int)
y_test = test_data['price']
metrics= model.evaluate(x_test, y_test)
print(f"Mean Squared Error: {metrics}")
