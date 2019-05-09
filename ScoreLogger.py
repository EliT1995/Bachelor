from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
import pandas

#SCORES_CSV_PATH = "./scores.csv"
#SCORES_PNG_PATH = "./scores.png"
#SOLVED_CSV_PATH = "./solved.csv"
#SOLVED_PNG_PATH = "./solved.png"

class ScoreLogger:

    def __init__(self):

        self.SOLVED_CSV_PATH1 = "./solved_{}.csv".format("CartPole-v0")
        self.SOLVED_CSV_PATH2 = "./solved_{}.csv".format("CartPole-v0_new")
        self.SOLVED_CSV_PATH3 = "./solved_{}.csv".format("CartPole-v0_multi")
        self.SOLVED_PNG_PATH = "./solved_{}.png".format("CartPole-v0_combined")

    def add_score(self):
        scores1 = self.get_average_scores(self.SOLVED_CSV_PATH1)
        scores2 = self.get_average_scores(self.SOLVED_CSV_PATH2)
        scores3 = self.get_average_scores(self.SOLVED_CSV_PATH3)

        self._save_png(input_path1=scores1, input_path2=scores2,
                       input_path3=scores3,
                       output_path=self.SOLVED_PNG_PATH,
                       x_label="trials",
                       y_label="steps before solve",
                       average_of_n_last=None,
                       show_goal=False,
                       show_trend=False,
                       show_legend=True)

    def _save_png(self, input_path1, input_path2, input_path3, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []

        for i in range(0, len(input_path1)):
            x.append(i)
            y.append(input_path1[i])

        plt.subplots()
        plt.plot(x, y, 'r', label="CartPole-v0")

        x = []
        y = []
        for i in range(0, len(input_path2)):
            x.append(i)
            y.append(input_path2[i])

        plt.plot(x, y, label="CartPole-v0_new")

        x = []
        y = []
        for i in range(0, len(input_path3)):
            x.append(i)
            y.append(input_path3[i])

        plt.plot(x, y, 'g', label="CartPole-v0_multi")

        plt.title("CartPole")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def get_average_scores(self, input_path):
        scores_average = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(1000):
                score = []
                for i in range(0, 10):
                    print(j + 1000*i)
                    score.append(int(data[j + 1000*i][0]))
                scores_average.append(int(mean(score)))

        return scores_average


if __name__ == "__main__":

    score_logger = ScoreLogger()
    score_logger.add_score()
