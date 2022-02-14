import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import numpy as np
qMRI_results = ["R1 [1/sec]", "R2s [1/sec]", "MTV [fraction]", "R2 [1/sec]", "MT [p.u.]"]
EXP_NUMBER = 2


def out_if_bounds(df):
    _R1df = df.loc[df["R1 [1/sec]"] > 8]
    # _R1df = _R1df[["ExpNum", "Iron type", "Lipid type", "R1 [1/sec]"]]

    _R2stardf = df.loc[(df["R2s [1/sec]"] > 1) | (df["R2s [1/sec]"] < -1) | (df["R2s [1/sec]"] == "Nan")]
    # _R2stardf = _R2stardf[["ExpNum", "Iron type", "Lipid type", "R2s [1/sec]"]]

    _MTVdf = df.loc[(df["MTV [fraction]"] > 1) | (df["MTV [fraction]"] < -1)]
    # _MTVdf = _MTVdf[["ExpNum", "Iron type", "Lipid type", "MTV [fraction]"]]

    _R2df = df.loc[(df["R2 [1/sec]"] > 50) | (df["R2 [1/sec]"] < 0)]
    # _R2df = _R2df[["ExpNum", "Iron type", "Lipid type", "R2 [1/sec]"]]

    _MTdf = df.loc[(df["MT [p.u.]"] > 5)]
    # _MTdf = _MTdf[["ExpNum", "Iron type", "Lipid type", "MT [p.u.]"]]
    err_df = pd.concat([_R1df, _R2stardf, _MTVdf, _R2df, _MTdf]).drop_duplicates().reset_index(drop=True)

def lipid_analyse(df):
    df = df.loc[df["[Protein](mg/ml)"] == 0]
    df = df.loc[df["[Fe] (mg/ml)"] == 0]
    df = df.loc[~df["ExpNum"].astype(str).str.contains("O")]#remove Oshrat's exps
    df = df.sort_values(by=["Lipid [fr]"])
    df_for_opo = df.loc[df["Lipid type"] == "PC_SM"]
    df.to_csv('results/lipid analyse table.csv')
    fig = plt.figure(figsize=(30, 20))
    df["Lipid type"] = df["Lipid type"].replace({"PC_Cholest":0, "PC_SM":1, "PC":2})

    colors = ['red', 'green',  'blue']
    dicti = {0:'r', 1:'g', 2:'b'}
    dicti1 = {0:'PC_Cholest', 1:'PC_SM', 2:'PC'}

    for i in range(len(qMRI_results)):
        for j in range(3):
            temp = df.loc[df["Lipid type"] == j]
            x_np = temp["Lipid [fr]"].to_numpy()
            y_np = temp[qMRI_results[i]].to_numpy()
            coef = np.polyfit(x_np, y_np, 1)
            poly1d_fn = np.poly1d(coef)
            plt.subplot(2, 3, i + 1)
            plt.xlabel("lipid", fontsize=20)
            plt.ylabel(qMRI_results[i], fontsize=18)
            plt.plot(x_np, poly1d_fn(x_np), dicti[j]+'--',label =dicti1[j])
            plt.legend()
        x_np = df["Lipid [fr]"].to_numpy()
        y_np = df[qMRI_results[i]].to_numpy()
        plt.scatter(x_np, y_np, c=df["Lipid type"], cmap=matplotlib.colors.ListedColormap(colors))
        plt.title("lipid analyse with {}".format(qMRI_results[i]), fontsize=25)
    fig.suptitle("PC_Cholest - red \n\n PC_SM - green \n\n  PC - blue",y=0.4,x = 0.8, fontsize=30)

    plt.savefig('results/lipid analyse1.png')
    return df_for_opo


def iron_analyse(df):
    df = df.loc[df["Lipid [fr]"] == 0]
    # df = df.loc[df["[Protein](mg/ml)"] == 0]
    df = df.loc[~df["Iron type"].astype(str).str.contains("A")]  # remove Oshrat's exps
    df.to_csv('results/iron analyse table.csv')
    df["Iron type"] = df["Iron type"].replace({"Fe2":0, "Ferritin":1, "Transferrin":2,"Fe3":3})
    colors = ['red', 'green', 'blue', 'yellow']
    dicti = {0:'r', 1:'g', 2:'b', 3:'y'}
    dicti1 = {0: 'Fe2', 1: 'Ferritin', 2: 'Transferrin', 3: "Fe3"}
    fig = plt.figure(figsize=(30, 20))
    for i in range(len(qMRI_results)):
        for j in range(4):
            temp = df.loc[df["Iron type"] == j]
            x_np = temp["[Fe] (mg/ml)"].to_numpy()
            y_np = temp[qMRI_results[i]].to_numpy()
            coef = np.polyfit(x_np, y_np, 1)
            poly1d_fn = np.poly1d(coef)
            plt.subplot(2, 3, i + 1)
            plt.xlabel("[Fe] (mg/ml)", fontsize=20)
            plt.ylabel(qMRI_results[i], fontsize=18)
            plt.plot(x_np, poly1d_fn(x_np), dicti[j] + '--', label=dicti1[j])
            plt.legend()
        x_np = df["[Fe] (mg/ml)"].to_numpy()
        y_np = df[qMRI_results[i]].to_numpy()
        plt.scatter(x_np, y_np, c=df["Iron type"], cmap=matplotlib.colors.ListedColormap(colors))
        plt.title("iron analyse with {}".format(qMRI_results[i]), fontsize=25)
    fig.suptitle("Fe2 - red \n\n Ferritin - green \n\n  Transferrin - blue\n\n Fe3 - yellow",y=0.4,x = 0.8, fontsize=30)
    plt.savefig('results/iron analyse1.png')

def plot_everything(df):
    df["[Fe] (mg/ml)"] /=30
    # newdf = df["[Fe] (mg/ml)", ]
    # x_normal = np.arange(0,0.3,0.01)
    # x = np.arange(0,1,0.01)
    fig = plt.figure(figsize=(30, 20))
    for i in range(len(qMRI_results)):
        plt.subplot(2, 3, i + 1)
        # sns.pairplot(df)
        sns.scatterplot(data=df.loc[df["[Fe] (mg/ml)"] != 0], x="[Fe] (mg/ml)", y=qMRI_results[i],
                        hue = "Iron type",marker=">")
        sns.scatterplot(data=df.loc[df["Lipid [fr]"] != 0], x="Lipid [fr]", y=qMRI_results[i], hue = "Lipid type", alpha = 0.5)
        plt.title("data analyse with {}".format(qMRI_results[i]), fontsize=25)
    plt.savefig('results/all.png')
    # print(x)

def apo_analyse(df,df_for_opo):
    df = df.loc[df["Iron type"] == "ApoTrans"]
    df = df.loc[(df["[Protein](mg/ml)"] > 0) & (df["Lipid [fr]"] > 0 ) ]
    df = df.sort_values(by=['Lipid [fr]'])


    df.to_csv('results/apo analyse table.csv')
    df_for_opo.to_csv('results/lipid cs_sm analyse table.csv')


    # pdf = matplotlib.backends.backend_pdf.PdfPages("./to_rona/output.pdf")

    if __name__ == '__main__':
        print("POP")
    # f = open("./to_rona/new1",'w
    #     df = pd.read_excel('/ems/elsc-labs/mezer-a/david.cohen/Desktop/Project/new_dataold.xlsx')
    #     df = df.loc[~df["ExpNum"].astype(str).str.contains("O")]#remove Oshrat's exps
    #
    #     # df_for_opo = lipid_analyse(df)
        # iron_analyse(df)
        # apo_analyse(df,df_for_opo)
        plot_everything(df)

    # df.toc
    # for res in qMRI_results2:
    #     _df+=(_df[["ExpNum","Iron type","Lipid type",res]])
    # _df.to_csv("./to_rona/new1")
    # f.close()
    # plt.savefig(f)
    # # print("R1 above 8\n",R1_df[["ExpNum","Iron type","Lipid [fr]","[Protein](mg/ml)","[Fe] (mg/ml)","R1 [1/sec]"]])
    #
    # tables_dict = dict(tuple(df.groupby('ExpNum')))
    # for i in range(1, 15):
    #     for res in qMRI_results:
    #         pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    #         pdf.close()
    #         plt.figure()
    #         tables_dict[i] = tables_dict[i].sort_values(by = res)
    #         x_as = np.arange(len(tables_dict[i].index))
    #         y_as = np.array(tables_dict[i][res])
    #         max = np.amax(y_as)
    #         min = np.amin(y_as)
    #         plt.plot(x_as, y_as, 'ro')
    #         plt.title("max = {:.2f}, min = {:.2f}, expNum - {:.2f}, qMRI = {}".format(max,min,i, res))
    #
