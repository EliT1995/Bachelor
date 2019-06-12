from statistics import mean
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv


def get_average_scores(input_path):
    scores_average = []
    scores_stdev = []
    with open(input_path, "r") as scores:
        reader = csv.reader(scores)
        data = list(reader)
        for j in range(1000):
            score = []
            for i in range(0, 10):
                score.append(int(data[j + 1000*i][0]))
            scores_stdev.append(score)
            scores_average.append(int(mean(score)))

    return scores_average, scores_stdev


class ScoreLogger:

    def __init__(self):

        self.solved_csv_paths = []

        self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v0"))
        #self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v02"))
        self.solved_csv_paths.append("./solved_{}.csv".format("LunarLander-v23"))
        self.solved_csv_paths.append("./solved_{}.csv".format("LunarLander-v25"))
        self.solved_csv_paths.append("./solved_{}.csv".format("LunarLander-v210"))
        self.solved_csv_paths.append("./solved_{}.csv".format("LunarLander-v220"))
        #self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v050"))
        #self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v0100"))
        #self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v0200"))
        #self.solved_csv_paths.append("./solved_{}.csv".format("CartPole-v0500"))
        self.solved_png_path = "./solved_{}.png".format("LunarLander-v2-1351020")

    def add_score(self):
        self._save_png(input_path=self.solved_csv_paths, output_path = self.solved_png_path,
                       x_label="episodes",
                       y_label="accumulated average reward",
                       average_of_n_last=None,
                       show_goal=False,
                       show_trend=False,
                       show_legend=True)

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):

        colors = ["#ed9b90", "#4286f4", "#f2593e", "#f1bb3e", "#d5ed38", "#5e2a47", "#27dd82", "#666317", "#5a42e5", "#f442a4"]
        for i in range(len(input_path)):
            scores, scores_stdev = get_average_scores(input_path[i])
            x = []
            y = []
            scores_mean = deque(maxlen=100)
            low_CI = []
            upper_CI = []
            for j in range(0, len(scores)):
                scores_mean.append(scores[j])
                if j % 100 == 0:
                    y.append(int(mean(scores_mean)))

            y = y[1:]

            index = 0
            for j in range(0, len(y)):
                index += 100
                x.append(index)

            plt.plot(x, y, lw=2, color=colors[i], alpha=1, label=input_path[i])
            # plt.fill_between(x, low_CI, upper_CI, color='#ed9b90', alpha=0.4)

            # plt.plot(x, y, 'r', label="one-step DQN")

        plt.title("CartPole")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="lower right")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    score_logger = ScoreLogger()
    score_logger.add_score()
