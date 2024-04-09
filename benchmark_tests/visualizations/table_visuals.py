import matplotlib.pyplot as plt
import csv

def read_csv(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    header = data[0]
    data = data[1:]
    return header, data


def create_visual_table(file_path):
    header, data = read_csv(file_path)
    fig, ax = plt.subplots()
    ax.axis("off")

    table = ax.table(cellText = data, colLabels = header, loc = "center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col = list(range(len(header))))

    plt.show()