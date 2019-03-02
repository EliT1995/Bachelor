from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

#SCORES_CSV_PATH = "./scores.csv"
#SCORES_PNG_PATH = "./scores.png"
#SOLVED_CSV_PATH = "./solved.csv"
#SOLVED_PNG_PATH = "./solved.png"

class ScoreLogger:

    def __init__(self):

        self.SOLVED_CSV_PATH1 = "./scores/solved_{}.csv".format("CartPole-v0")
        self.SOLVED_CSV_PATH2 = "./scores/solved_{}.csv".format("CartPole-v0_new")
        self.SOLVED_CSV_PATH3 = "./scores/solved_{}.csv".format("CartPole-v0_multi")
        self.SOLVED_PNG_PATH = "./scores/solved_{}.png".format("CartPole-v0_combined")

    def add_score(self):
            self._save_png(input_path1=self.SOLVED_CSV_PATH1, input_path2=self.SOLVED_CSV_PATH2, input_path3=self.SOLVED_CSV_PATH3,
                           output_path=self.SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)

    def _save_png(self, input_path1, input_path2, input_path3, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path1, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, 'r')

        x = []
        y = []
        with open(input_path2, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.plot(x, y)

        x = []
        y = []
        with open(input_path3, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.plot(x, y, 'g')

        plt.title("CartPole")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":

    score_logger = ScoreLogger()
    score_logger.add_score()
