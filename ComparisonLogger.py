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
        SOLVED_CSV_PATH_Final = "./solved_{}.csv".format(self.name)

        final = []

        mean_score_final = 0

        SOLVED_CSV_PATH_i = "./scores_{}/solved_{}.csv".format(self.name, self.name + str(0))

        with open(SOLVED_CSV_PATH_i, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for j in range(0, len(data)):
                final.append(int(data[j][0]))
                mean_score_final += int(data[j][0])

        for i in range(1, 100):
            mean_score = 0
            final_scores = []
            SOLVED_CSV_PATH_i = "./scores_{}/solved_{}.csv".format(self.name, self.name + str(i))
            with open(SOLVED_CSV_PATH_i, "r") as scores:
                reader = csv.reader(scores)
                data = list(reader)
                for j in range(0, len(data)):
                    final_scores.append(int(data[j][0]))
                    mean_score += int(data[j][0])
                if mean_score > mean_score_final:
                    mean_score_final = mean_score
                    final = final_scores

        for i in range(len(final)):
            self._save_csv(SOLVED_CSV_PATH_Final, int(final[i]))