import matplotlib.pyplot as plt
import pandas as pd
import csv

def read_csv(file_path, exclude = ["true optimum"]):
    df = pd.read_csv(file_path, index_col=0)
    return df

#track_type,mpso_type,function_name,average_time,max_time,min_time,std-dev_time,average_g_best_value,max_g_best_value,min_g_best_value,std-dev_g_best_value,true bias,true optimum,average_time_ccd,max_time_ccd,min_time_ccd,std-dev_time_ccd,average_g_best_value_ccd,max_g_best_value_ccd,min_g_best_value_ccd,std-dev_g_best_value_ccd


def create_statistics(df):
    print(df)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    print(numeric_columns)
    statistics = {
        "Average": df[numeric_columns].mean(),
        "Min": df[numeric_columns].min(),
        "Max": df[numeric_columns].max(),
        "Std Dev": df[numeric_columns].std()
        }
    return statistics

def create_table_image(statistics):
    fig, ax = plt.subplots()
    ax.axis("off")

    table_data = []
    for column, stats in statistics.items():
        row = [column] + [f"{value:.2f}" for value in stats.values]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels = ["Statistic", *statistics["Average"].index], loc = "center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.savefig('statistics_table.png', bbox_inches='tight')
    plt.show()

def display_table(file_path):
    df = read_csv(file_path)
    print(df)
    statistics = create_statistics(df)
    print(statistics)
    create_table_image(statistics)