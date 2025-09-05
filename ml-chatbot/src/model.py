class Model:
    def __init__(self):
        self.model = None

    def build_model(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()

    def train_model(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model is not built. Call build_model() first.")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise Exception("Model is not built. Call build_model() first.")
        return self.model.predict(X)