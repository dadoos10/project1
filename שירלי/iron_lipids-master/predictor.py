from plots import *


class Predictor:

    def __init__(self, X, y, data):
        """ initiate the Predictor instance """
        # build linear regression model
        self.model = LinearRegression()
        # define cross-validation method to use
        self.cross_validation = LeaveOneOut()
        self.X = X
        self.y = y
        self.mae = 0
        self.accuracy = 0
        self.data = data

    def scores(self):
        """ the function calculates the scores of the model after cross validation"""
        scores = cross_val_score(self.model, self.X, self.y, scoring='neg_mean_squared_error',
                                 cv=self.cross_validation, n_jobs=-1)
        self.accuracy = r2_score(self.y, self.predict())
        self.mae = mean(absolute(scores))

    def predict(self):
        """ the function returns predictions of the target based on cross validation linear
        regression"""
        return cross_val_predict(self.model, self.X, self.y, cv=self.cross_validation)

    def plot(self,target_name):
        """ the function create instance of Plot class and plots the data """
        self.scores()
        title = "R^2: " + str(float("{:.3f}".format(self.accuracy))) + " Mean absolute squared " \
                "error: " + str(float("{:.3f}".format(self.mae)))
        norm = np.linalg.norm(self.y)
        self.y /= norm
        # self.y = (self.y - self.y.mean()) / self.y.std()

        res = self.predict()
        norm = np.linalg.norm(res)
        res /= norm
        # res = (res - res.mean()) / res.std()

        pl = Plots(self.y, res, target_name+" measured", target_name+" predicted", title, self.data)
        pl.plot()
