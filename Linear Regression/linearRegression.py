import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
# y = w_1 * x_1 + w_2 * x_2 + .... + w_n * x_n + b

'''
Steps:
1) Initialize a weight matrix of shape (x_rows, x_cols) 
2) Initialize a bias matrix of shape (x_rows, 1)
3) Define cost function using MSE.
4) Perform Gradient Descent on the cost function.
5) Update weights and biases.
6) Reiterate
7) After training is complete, take column-wise mean of all columns for weights and biases.
'''

class LinearRegression:
    weight_matrix = None
    bias_matrix = None
    columns = None
    alpha = None
    epochs = None
    x_train = None
    y_train = None
    train_shape = None
    error = None

    def __init__(self, x_train, y_train, epochs=20, alpha=0.001):
        self.x_train = pd.DataFrame(x_train)
        self.x_train /= self.x_train.max()
        self.y_train = pd.DataFrame(y_train)
        self.y_train /= self.y_train.max()
        self.shape = x_train.shape
        self.y_train = np.resize(self.y_train, (self.shape[0], 1))
        self.weight_matrix = np.random.rand(self.shape[0], self.shape[1])
        self.bias_matrix = np.random.rand(self.shape[0], 1)
        self.epochs = epochs
        self.alpha = alpha
        self.error = 0


    def train(self):
        for i in range(self.epochs):
            #print(np.sum(np.matmul(self.weight_matrix, self.x_train.T), axis=1))
            predictions = np.resize(np.sum(np.matmul(self.weight_matrix, self.x_train.T), axis=1), (self.shape[0], 1)) + self.bias_matrix

            error_matrix = ((self.y_train - predictions) ** 2) / self.shape[0] ## MSE Matrix
            self.error = np.sum(error_matrix, axis=1) ## Cumulative MSE

            self.weight_matrix = self.weight_matrix - ((self.alpha * 2 * (predictions - self.y_train) * self.weight_matrix) / self.shape[0]) ## weight update
            self.bias_matrix = self.bias_matrix - ((self.alpha * 2 * (predictions - self.y_train) * self.bias_matrix) / self.shape[0])

            if i % 10 ==0:
                print(self.error.mean())
        self.weight_matrix = np.mean(self.weight_matrix, axis=0).reshape(1, -1)
        self.bias_matrix = np.mean(self.bias_matrix, axis=1).reshape(1, 1)
        print()
        print("*" * 50)
        print("Training complete.")
        print("*" * 50)
        return self.error.mean()

    def predict(self, x_features):
        # Convert to DataFrame and normalize with training max values
        x_features = pd.DataFrame(x_features) / self.x_train.max()

        # Predict values
        y_predictions = np.matmul(x_features.values, self.weight_matrix.T) + self.bias_matrix  # Dot product with shape alignment
        return y_predictions.flatten()  # Flatten to (n_samples,)

    def evaluate(self, x_test, y_test):
        # Generate predictions and reshape y_test for compatibility
        y_predictions = self.predict(x_test)
        y_test = pd.Series(y_test).values  # Convert y_test to numpy array

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_test - y_predictions) ** 2)

        # Define accuracy based on a threshold
        threshold = 0.1  # Define acceptable error threshold
        accuracy = np.mean(np.abs((y_test - y_predictions) / y_test) < threshold) * 100

        return {"Mean Squared Error": mse, "Accuracy (%)": accuracy}


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
error = model.train()
test_data = pd.DataFrame(pd.read_csv("Dataset/House price/df_test.csv"))
x_test = test_data.drop(columns=['price'])
x_test["has_basement"] = x_test["has_basement"].replace({True:1, False:0}).astype(int)
x_test["renovated"] = x_test["renovated"].replace({True: 1, False: 0}).astype(int)
x_test["nice_view"] = x_test["nice_view"].replace({True:1, False:0}).astype(int)
x_test["perfect_condition"] = x_test["perfect_condition"].replace({True:1, False:0}).astype(int)
x_test["has_lavatory"] = x_test["has_lavatory"].replace({True:1, False:0}).astype(int)
x_test["single_floor"] = x_test["single_floor"].replace({True:1, False:0}).astype(int)
y_test = test_data['price']

print(error)
print(model.evaluate(x_test, y_test))
