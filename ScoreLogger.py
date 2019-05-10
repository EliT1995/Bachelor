from statistics import mean
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
from ComparisonLogger import ComparisonLogger


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

        self.SOLVED_CSV_PATH1 = "./solved_{}.csv".format("CartPole-v0")
        self.SOLVED_CSV_PATH2 = "./solved_{}.csv".format("CartPole-v0_2step")
        self.SOLVED_CSV_PATH3 = "./solved_{}.csv".format("CartPole-v0_new")
        self.SOLVED_CSV_PATH4 = "./solved_{}.csv".format("CartPole-v0_10step")
        self.SOLVED_PNG_PATH = "./solved_{}.png".format("CartPole-v0_12310")

    def add_score(self):
        self._save_png(input_path1=self.SOLVED_CSV_PATH1, input_path2=self.SOLVED_CSV_PATH2,
                       input_path3=self.SOLVED_CSV_PATH3, input_path4=self.SOLVED_CSV_PATH4,
                       output_path=self.SOLVED_PNG_PATH,
                       x_label="episodes",
                       y_label="accumulated average reward",
                       average_of_n_last=None,
                       show_goal=False,
                       show_trend=False,
                       show_legend=True)

    def _save_png(self, input_path1, input_path2, input_path3, input_path4, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        scores1, scores1_stdev = get_average_scores(self.SOLVED_CSV_PATH1)
        scores2, scores2_stdev = get_average_scores(self.SOLVED_CSV_PATH2)
        scores3, scores3_stdev = get_average_scores(self.SOLVED_CSV_PATH3)
        scores4, scores4_stdev = get_average_scores(self.SOLVED_CSV_PATH4)

        x = []
        y = []
        scores = deque(maxlen=50)
        low_CI = []
        upper_CI = []
        for i in range(0, len(scores1)):
            scores.append(scores1[i])
            y.append(int(scores1[i]))

        for i in range(len(scores1_stdev)):
            scores_np = np.array(scores1_stdev[i])
            stdev = np.std(scores_np)
            low_CI.append(int(y[i] - stdev))
            upper_CI.append(int(y[i] + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#ed9b90', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#ed9b90', alpha=0.4, label="one-step DQN")

        #plt.plot(x, y, 'r', label="one-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        low_CI = []
        upper_CI = []
        for i in range(0, len(scores2)):
            scores.append(scores2[i])
            y.append(int(scores2[i]))

        for i in range(len(scores2_stdev)):
            scores_np = np.array(scores2_stdev[i])
            stdev = np.std(scores_np)
            low_CI.append(int(y[i] - stdev))
            upper_CI.append(int(y[i] + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#539caf', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#539caf', alpha=0.4, label="two-step DQN")
        #plt.plot(x, y, label="three-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        low_CI = []
        upper_CI = []
        for i in range(0, len(scores3)):
            scores.append(scores3[i])
            y.append(int(scores3[i]))

        for i in range(len(scores3_stdev)):
            scores_np = np.array(scores3_stdev[i])
            stdev = np.std(scores_np)
            low_CI.append(int(y[i] - stdev))
            upper_CI.append(int(y[i] + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#eaef8f', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#eaef8f', alpha=0.4, label="three-step DQN")
        #plt.plot(x, y, 'g', label="three one-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        low_CI = []
        upper_CI = []
        for i in range(0, len(scores4)):
            scores.append(scores4[i])
            y.append(int(scores4[i]))

        for i in range(len(scores4_stdev)):
            scores_np = np.array(scores4_stdev[i])
            stdev = np.std(scores_np)
            low_CI.append(int(y[i] - stdev))
            upper_CI.append(int(y[i] + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#db15d4', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#db15d4', alpha=0.4, label="ten-step DQN")

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
