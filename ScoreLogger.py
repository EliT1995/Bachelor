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
    with open(input_path, "r") as scores:
        reader = csv.reader(scores)
        data = list(reader)
        for j in range(1000):
            score = []
            for i in range(0, 10):
                score.append(int(data[j + 1000*i][0]))
            scores_average.append(int(mean(score)))

    return scores_average


class ScoreLogger:

    def __init__(self):

        self.SOLVED_CSV_PATH1 = "./solved_{}.csv".format("CartPole-v0_200step")
        self.SOLVED_CSV_PATH2 = "./solved_{}.csv".format("CartPole-v0_500step")
        self.SOLVED_CSV_PATH3 = "./solved_{}.csv".format("CartPole-v0_1000step")
        self.SOLVED_PNG_PATH = "./solved_{}.png".format("CartPole-v0_probe")

    def add_score(self):
        scores1 = get_average_scores(self.SOLVED_CSV_PATH1)
        scores2 = get_average_scores(self.SOLVED_CSV_PATH2)
        scores3 = get_average_scores(self.SOLVED_CSV_PATH3)

        self._save_png(input_path1=scores1, input_path2=scores2,
                       input_path3=scores3,
                       output_path=self.SOLVED_PNG_PATH,
                       x_label="episodes",
                       y_label="accumulated average reward",
                       average_of_n_last=None,
                       show_goal=False,
                       show_trend=False,
                       show_legend=True)

    def _save_png(self, input_path1, input_path2, input_path3, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        scores = deque(maxlen=50)
        logger = ComparisonLogger(self.SOLVED_CSV_PATH1)
        low_CI = []
        upper_CI = []
        for i in range(0, len(input_path1)):
            scores.append(input_path1[i])
            #y.append(int(input_path1[i]))
            if len(scores)>=50:
                scores_np = np.array(scores)
                stdev = np.std(scores_np)
                y.append(int(mean(scores_np)))
                low_CI.append(int(mean(scores_np) - stdev))
                upper_CI.append(int(mean(scores_np) + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#ed9b90', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#ed9b90', alpha=0.4, label="200-step DQN")

        #plt.plot(x, y, 'r', label="one-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        logger = ComparisonLogger(self.SOLVED_CSV_PATH2)
        low_CI = []
        upper_CI = []
        for i in range(0, len(input_path2)):
            scores.append(input_path2[i])
            # y.append(int(input_path1[i]))
            if len(scores) >= 50:
                scores_np = np.array(scores)
                stdev = np.std(scores_np)
                y.append(int(mean(scores_np)))
                low_CI.append(int(mean(scores_np) - stdev))
                upper_CI.append(int(mean(scores_np) + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#539caf', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#539caf', alpha=0.4, label="500-step DQN")
        #plt.plot(x, y, label="three-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        logger = ComparisonLogger(self.SOLVED_CSV_PATH3)
        low_CI = []
        upper_CI = []
        for i in range(0, len(input_path3)):
            scores.append(input_path3[i])
            # y.append(int(input_path1[i]))
            if len(scores) >= 50:
                scores_np = np.array(scores)
                stdev = np.std(scores_np)
                y.append(int(mean(scores_np)))
                low_CI.append(int(mean(scores_np) - stdev))
                upper_CI.append(int(mean(scores_np) + stdev))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, lw=2, color='#eaef8f', alpha=1)
        plt.fill_between(x, low_CI, upper_CI, color='#eaef8f', alpha=0.4, label="1000-step DQN")
        #plt.plot(x, y, 'g', label="three one-step DQN")

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
