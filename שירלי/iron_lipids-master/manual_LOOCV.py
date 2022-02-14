""" calculate manual LOO cross-validation """

from toolBox import *


def pre_processing():
    """
    The function pre process the data, get the data from the input file, remove bad samples and
    prepare the data according to chosen lipids and proteins types.
    :return:
    """
    data = pd.read_excel(PATH)
    # ignore experiments 6
    for bad_sample in bad_samples:
        data = data[data.ExpNum != bad_sample]
    data = data[data[LIPID] != 0.0]
    highest = data[R1].mean() + 3 * data[R1].std()
    lowest = data[R1].mean() - 3 * data[R1].std()

    data = data[(data[R1] < highest) & (data[R1] > lowest)]
    return data


def manual_loo(data):

    X, y = predict_R1(data)
    # X = standardize(X)
    cv = LeaveOneOut()
    # enumerate splits
    y_true, y_pred, coeff_iron_Fe2, coeff_iron_Ferritin, coeff_iron_Trans, coeff_lipid_Pc, \
    coeff_lipid_PC_Ch, coeff_lipid_PC_SM, scores = list(), list(), list(), list(),  list(), \
                                                   list(), list(), list(), list()
    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # evaluate model
        yhat = model.predict(X_test)

        # store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
        # store score

        coeff_iron_Fe2.append(model.coef_[0])
        coeff_iron_Ferritin.append(model.coef_[1])
        coeff_iron_Trans.append(model.coef_[2])
        coeff_lipid_Pc.append(model.coef_[3])
        coeff_lipid_PC_Ch.append(model.coef_[4])
        coeff_lipid_PC_SM.append(model.coef_[5])
        print(model.coef_)

    scores = r2_score(y_true, y_pred)
    # summarize model performance
    mean_s, std_s = mean(scores), np.std(scores)
    print('Mean: {0} Standard Deviation: {1}'.format(mean_s, std_s))

    plt.scatter([i for i in range(0, len(coeff_iron_Fe2))], coeff_iron_Fe2, label="Fe2")
    plt.scatter([i for i in range(0, len(coeff_iron_Ferritin))], coeff_iron_Ferritin, label="Ferritin")
    plt.scatter([i for i in range(0, len(coeff_iron_Trans))], coeff_iron_Trans, label="Trans")
    plt.scatter([i for i in range(0, len(coeff_lipid_Pc))], coeff_lipid_Pc, label="PC")
    plt.scatter([i for i in range(0, len(coeff_lipid_PC_Ch))], coeff_lipid_PC_Ch,
                label="PC_Cholest")
    plt.scatter([i for i in range(0, len(coeff_lipid_PC_SM))], coeff_lipid_PC_SM, label="PC_SM")

    print(mean(coeff_iron_Fe2))
    print(mean(coeff_iron_Ferritin))
    print(mean(coeff_iron_Trans))
    print(mean(coeff_lipid_Pc))
    print(mean(coeff_lipid_PC_Ch))
    print(mean(coeff_lipid_PC_SM))

    plt.xlabel("Iteration number")
    plt.ylabel("standardized coeff")
    plt.legend()
    plt.show()


def predict_R1(data):
    X = data[[IRON, IRON_TYPE, LIPID, LIPID_TYPE]]
    # get numeric representation of lipid types
    X = pd.get_dummies(data=X, drop_first=False)

    for col in X.columns[2:5]:
        X[col] = X[col] * X[IRON]
    for col in X.columns[5:8]:
        X[col] = X[col] * X[LIPID]
    print(X[X.columns[2:]].head())
    y = np.array(data[R1])
    X = np.array(X[X.columns[2:]]).reshape(-1, 6)

    for i in range(len(X[0])):
        X[:, i] = scale(X[:, i])

    return X, y

if __name__ == '__main__':
    manual_loo(pre_processing())