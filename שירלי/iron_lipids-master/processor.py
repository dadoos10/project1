from toolBox import *


class Processor:

    def __init__(self, target,data):
        self.data = data
        self.target = target

    # def pre_processing(self):
    #     """
    #     The function pre process the data, get the data from the input file, remove bad samples and
    #     prepare the data according to chosen lipids and proteins types.
    #     :return:
    #     """
    #     # create our independent variables
    #     X = self.data[[IRON, IRON_TYPE, LIPID, LIPID_TYPE]]
    #     # get numeric representation of lipid and iron types
    #     X = pd.get_dummies(data=X, drop_first=False)
    #
    #     for col in X.columns[2:7]:
    #         X[col] = X[col] * X[IRON]
    #     for col in X.columns[8:10]:
    #         X[col] = X[col] * X[LIPID]
    #     # insert types as multiplication
    #     # ignore experiments 6
    #     for bad_sample in bad_samples:
    #         self.data = self.data[self.data.ExpNum != bad_sample]
    #     self.data = self.data[self.data[LIPID] != 0.0]

    def get_data(self):
        return self.data

    def relations_target(self, for_hue, target, relation_sub):
        # Compute the correlation matrix
        corr = self.data.corr()
        # print(corr[target])
        #corr between target and concentration
        print(stats.pearsonr(self.data[target], self.data[relation_sub]))
        #2 types of plots for 2 types of sets
        g = sns.pairplot(self.data, hue=for_hue, palette="muted", size=5,
                         vars=[target, relation_sub], kind='reg')

        # remove the top and right line in graph
        sns.despine()
        # Additional line to adjust some appearance issue
        plt.subplots_adjust(top=0.9)

        plt.show()

    def detect_outliers(self):
        import warnings
        #show target condition in box plot
        # sns.boxplot(x=self.data[self.target])
        # plt.show()

        #get rid of outliers
        highest = self.data[self.target].mean() + 3 * self.data[self.target].std()
        lowest = self.data[self.target].mean() - 3 * self.data[self.target].std()
        self.data = self.data[(self.data[self.target] < highest) & (self.data[self.target] > lowest)]


        # sns.boxplot(x=self.data[self.target])
        # plt.show()