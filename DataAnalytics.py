import matplotlib.pyplot as plt

import pickle
from Experiment import *
import os
import statistics
import pandas as pd

class DataAnalytics:
    def __init__(self):
        files = os.listdir(Experiment.DIR_NAME)
        data_folder = Path(Experiment.DIR_NAME)

        self.experiments = []
        for f in files:
            f = open(data_folder / f, "rb")
            experiment = pickle.load(f)
            while 1:
                try:
                    experiment.testResults.update(pickle.load(f))
                except EOFError:
                    break
            self.experiments.append(experiment)

    def drawChart(self, property, constants, xLabel, title):
        filtered = []
        AVERAGE = "average"
        BEST = "best"
        WORST = "worst"

        for exp in self.experiments:
            keep = False
            for key, value in constants.items():
                if getattr(exp, key) == value:
                    keep = True
                    break
            if keep:
                cp = {}
                cp[property] = getattr(exp, property)
                cp[AVERAGE] = exp.calculateAverage()
                cp[BEST] = exp.calculateBest()
                cp[WORST] = exp.calculateWorst()
                filtered.append(cp)


        df = pd.DataFrame(filtered, columns = filtered[0].keys())
        df_avg = df.groupby(property)[AVERAGE].agg('mean').reset_index()
        df_best = df.groupby(property)[BEST].agg('min').reset_index()
        df_worst = df.groupby(property)[WORST].agg('max').reset_index()
        plt.plot(property, AVERAGE, data= df_avg, marker='', color='olive', linewidth=2, linestyle='dashed', label="Average")
        plt.plot(property, BEST, data= df_best, marker='', color='blue', linewidth=2, linestyle='solid', label="Best")
        plt.plot(property, WORST, data= df_worst, marker='', color='red', linewidth=2, linestyle='solid', label="Worst")

        img_folder = Path("charts")
        filename = title+ ".png"
        file_to_save = img_folder / filename
        plt.savefig(file_to_save)

        plt.legend()
        plt.xlabel(xLabel)
        plt.ylabel('Fitness')
        plt.show()

        def drawChartByMutationType(self, property, constants, xLabel, title):
            filtered = []
            AVERAGE = "average"
            BEST = "best"
            WORST = "worst"

            for exp in self.experiments:
                keep = False
                for key, value in constants.items():
                    if getattr(exp, key) == value:
                        keep = True
                        break
                if keep:
                    cp = {}
                    cp[property] = getattr(exp, property)
                    cp[AVERAGE] = exp.calculateAverage()
                    cp["mutation"] = str(exp.mutation)
                    filtered.append(cp)


            df = pd.DataFrame(filtered, columns = filtered[0].keys())
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, df in df.groupby("mutation"):
                df.average.plot(kind="line", ax=ax, label=label)
            plt.legend()
            plt.xlabel(xLabel)
            plt.ylabel('Fitness')
            plt.show()

