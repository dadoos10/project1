import matplotlib.pyplot as plt
import numpy as np

from processor import *
from predictor import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics\
    import mean_absolute_error
iron_dicti = {"Fe2":0, "Ferritin":1, "Transferrin":2, "Fe3":3,"Ferr+Trans":4 }
lip_dicti = {"PC_Cholest":0, "PC":1, "PC_SM":2}

def categorical_cross_val(data, target):
    """
    This function performs cross validation using 'leave one out' method.
    The function predicts the target variable using one or several predictors.
    plots only one chosen model
    """
    # create our independent variables
    #normelize data
    # for i in ["[Protein](mg/ml)","iron [mg/ml]","lipid [%]"
    #     ,"R1 [1/sec]","R2s [1/sec]","MTV [fraction]",
    #           "R2 [1/sec]","MT [p.u.]","[Fe] sigma [mg/ml]",
    #           "[Fe] estimation [mg/ml]"]:
    #     data[i] = (data[i] - data[i].mean()) /data[i].std()

    X = data[[R1,R2,R2S,MTV,MT]]
    # get numeric representation of lipid and iron types

    # insert types as multiplication
    y = np.array(data[target])
    # X = np.array(X).reshape(-1, 6)

    # # normalize the data
    # for i in range(len(X[0])):
    #     X[:, i] = scale(X[:, i])

    pred = Predictor(X, y, data)
    pred.plot(target)


def cross_val(data, target):
    models = {
        'IRON': np.array(data[IRON]).reshape(-1, 1),
        'LIPID': np.array(data[LIPID]).reshape(-1, 1),
        'IRON, LIPID': np.array(data[[IRON, LIPID]]).reshape(-1, 2),
        'IRON, LIPID, IRON * LIPID': np.hstack((np.array(data[[IRON, LIPID]]).reshape(-1, 2),
                                                np.array(data[IRON] * data[LIPID]).reshape(
                                                    -1, 1))),
        'IRON * LIPID': np.array(data[IRON] * data[LIPID]).reshape(-1, 1),
    }

    for model in models:
        # create X, y
        X = models[model]
        y = np.array(data[target])
        # normalize the data
        for i in range(len(X[0])):
            X[:, i] = scale(X[:, i])

        pred = Predictor(X, y, data)
        pred.plot(target)

def target_Pre(data):
    data = data.dropna()
    #drop problematic exp
    bad_exp = [6,"O1","O2","O3"]
    for i in bad_exp:
        data = data[data.ExpNum != i]
    data = data[data[LIPID] != 0.0]
    # make types to categorical according to dictionaries.
    data[IRON_TYPE] = data[IRON_TYPE].replace(iron_dicti)
    data[LIPID_TYPE] = data[LIPID_TYPE].replace(lip_dicti)
    data[IRON_TYPE] = data[IRON_TYPE][data[IRON_TYPE] != "ApoTrans"]
    data = data.dropna()
    # ignore protein param for now..
    del data[PROTEIN]


    # take just one lipid type, and check lipid q
    # iron_dicti = {"Fe2": 0, "Ferritin": 1, "Transferrin": 2, "Fe3": 3, "Ferr+Trans": 4}

    data = data.loc[data[IRON_TYPE] == 1]
    # data = data.loc[data[IRON] == 0]
    # data = data.loc[data["ExpNum"] == 4]
    # data = data.loc[data["ExpNum"] != 4]
    # data[R2S] = data[R
    #  2]*data[R2S]
    targeted = data[[IRON, IRON_TYPE, LIPID, LIPID_TYPE]]
    return data[[MT, ]],targeted , targeted.columns
    # return data[[R2,R2S,MTV, MT]],targeted , targeted.columns

if __name__ == '__main__':
    data = pd.read_excel(PATH)
    prepro_data, target_data, targets = target_Pre(data)
    # print(prepro_data)
    # print(target_data)
    _targets = [LIPID]
    # _targets = [IRON, IRON_TYPE, LIPID, LIPID_TYPE]
    X = prepro_data
    for target in _targets:
        y = target_data[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=2)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        print(target+" values are:")
        print("score: %.5f"%(regressor.score(X_test,y_test, sample_weight=None)))
        #values of w vector
        print("w :", list(np.around(regressor.coef_,5)))
        print("intercept :", np.around(regressor.intercept_,5))
        print("MSE: %.5f"%mean_squared_error(regressor.predict(X_test),y_test))


        plt.plot(y_test, regressor.predict(X_test),'o')
        plt.plot([0, 1])
        plt.plot()
        plt.show()
