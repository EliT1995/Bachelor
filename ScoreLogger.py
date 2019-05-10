from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import csv


def get_average_scores(input_path):
    scores_average = []
    with open(input_path, "r") as scores:
        reader = csv.reader(scores)
        data = list(reader)
        for j in range(1000):
            score = []
            for i in range(0, 8):
                score.append(int(data[j + 1000*i][0]))
            scores_average.append(int(mean(score)))

    return scores_average


class ScoreLogger:

    def __init__(self):

        self.SOLVED_CSV_PATH1 = "./solved_{}.csv".format("CartPole-v0")
        self.SOLVED_CSV_PATH2 = "./solved_{}.csv".format("CartPole-v0_new")
        self.SOLVED_CSV_PATH3 = "./solved_{}.csv".format("CartPole-v0_multi")
        self.SOLVED_PNG_PATH = "./solved_{}.png".format("CartPole-v0_combined")

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
        for i in range(0, len(input_path1)):
            scores.append(input_path1[i])
            if len(scores)>=50:
                y.append(int(mean(scores)))

        for i in range(0, len(y)):
            x.append(i)

        plt.subplots()
        plt.plot(x, y, 'r', label="one-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        for i in range(0, len(input_path2)):
            scores.append(input_path2[i])
            if len(scores) >= 50:
                y.append(int(mean(scores)))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, label="three-step DQN")

        x = []
        y = []
        scores = deque(maxlen=50)
        for i in range(0, len(input_path3)):
            scores.append(input_path3[i])
            if len(scores) >= 50:
                y.append(int(mean(scores)))

        for i in range(0, len(y)):
            x.append(i)

        plt.plot(x, y, 'g', label="three one-step DQN")

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
