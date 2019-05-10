import csv
import os


class ComparisonLogger:

    def __init__(self, name):
        self.name = name

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    def get_best_score(self):
        SOLVED_CSV_PATH_Final = "./{}".format(self.name)

        final_max = []
        mean_score_final = 0

        SOLVED_CSV_PATH_i = "./{}".format(self.name)

        with open(SOLVED_CSV_PATH_i, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(0, 1000):
                final_max.append(int(data[j][0]))
                mean_score_final += int(data[j][0])

        with open(SOLVED_CSV_PATH_i, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(1,10):
                mean_score = 0
                final_scores = []
                for i in range(1000):
                    final_scores.append(int(data[i + 1000*j][0]))
                    mean_score += int(data[i + 1000*j][0])
                if mean_score > mean_score_final:
                    mean_score_final = mean_score
                    final_max = final_scores

        return final_max

    def get_worst_score(self):
        SOLVED_CSV_PATH_Final = "./{}".format(self.name)

        final_min = []
        mean_score_final = 0

        SOLVED_CSV_PATH_i = "./{}".format(self.name)

        with open(SOLVED_CSV_PATH_i, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(0, 1000):
                final_min.append(int(data[j][0]))
                mean_score_final += int(data[j][0])

        with open(SOLVED_CSV_PATH_i, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(1,10):
                mean_score = 0
                final_scores = []
                for i in range(1000):
                    final_scores.append(int(data[i + 1000*j][0]))
                    mean_score += int(data[i + 1000*j][0])
                if mean_score < mean_score_final:
                    mean_score_final = mean_score
                    final_min = final_scores

        return final_min