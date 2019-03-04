from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

CONSECUTIVE_RUNS_TO_SOLVE = 100


class StatistikLogger:

    def __init__(self, env_name, threshold):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.scoresWindow = deque(maxlen=20)
        self.env_name = env_name
        self.AVERAGE_SCORE_TO_SOLVE = threshold

        self.SOLVED_CSV_PATH = "./solved_{}.csv".format(env_name)
        self.SOLVED_PNG_PATH = "./solved_{}.png".format(env_name)

        if os.path.exists(self.SOLVED_PNG_PATH):
            os.remove(self.SOLVED_PNG_PATH)

    def add_score(self, score, run):
        self.score = score
        self.run = run

        self.scores.append(score)
        mean_score = round(mean(self.scores))
        self.scoresWindow.append(mean_score)
        print("Run: {}, Step: {}, Score: (min: {}, avg: {}, max: {})".format(run, self.score, min(self.scores), mean_score, max(self.scores)))

        if len(self.scoresWindow) == 20:
            solve_score = int(mean(self.scoresWindow))
            self._save_csv(self.SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=self.SOLVED_CSV_PATH,
                           output_path=self.SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before dying",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=True,
                           show_legend=False)

        if mean_score >= self.AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            print("Solved in {} runs {} total runs".format(solve_score, run))
            #exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [self.AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(self.AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])