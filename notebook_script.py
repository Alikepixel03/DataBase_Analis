from random import Random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import moviepy.editor as mpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import sklearn

print(sklearn.__version__)

df = pd.read_csv("vgsales.csv", sep=",")

print(df.head())

print(df.info())

print(df.tail())

print(df.Platform.unique())

print(df.Year.unique())

print(df.Genre.unique())

df.Global_Sales.plot()
plt.show()

print(df[df.Global_Sales > 80])

print(df[df.Global_Sales < 1].head())

print(len(df[df.Global_Sales < 1]))

def plot_chart(i):
    df.Global_Sales.plot()
    plt.ylim([0, i])

for i in range(80, 0, -1):
    fig = plt.figure()
    plot_chart(i)
    fig.savefig(f"figures/{i}.png")
    plt.close()

figures_lst = [f"figures/{i}.png" for i in range(80, 10, -1)]

for i in range(10, 0, -1):
    for k in range((11 - i) * 2):
        figures_lst.append(f"figures/{i}.png")

GIF_NAME = "global_sales.gif"
FPS = 0.01
clip = mpy.ImageSequenceClip(figures_lst, fps=FPS)
clip.write_gif(f"{GIF_NAME}", fps=FPS)

# Код Markdown для отображения .gif: <img src="global_sales.gif" width="420" align="center">

print(
    df[df.Publisher == "Electronic Arts"]
    .sort_values(by=["Global_Sales"], ascending=False)
    .head()
)

print(df.Year.dropna().unique().min())
print(df.Year.dropna().unique().max())

print(df[df.Year > 2017].head())

years_sales = dict()
print("Global_Sales:")

for year in list(range(1980, 2018)) + [2020]:
    years_sales[year] = df[df.Year == year].iloc[0].Global_Sales

    print(f"{year}:", df[df.Year == year].iloc[0].Global_Sales)

print(
    pd.DataFrame(
        {
            k: v
            for k, v in sorted(years_sales.items(), key=lambda v: v[1], reverse=True)
        },
        index=["Global_Sales"],
    ).T.head()
)

print(df.isnull().sum())

df_work = df.copy()

df_work.Year.fillna(df.Year.median(), inplace=True)

df_work.dropna(inplace=True)

df_work.dropna(inplace=True)

print(df_work.isnull().sum())

platform_values = list(df_work.Platform.unique())
genre_values = list(df_work.Genre.unique())
publisher_values = list(df_work.Publisher.unique())

print(len(platform_values))
print(len(genre_values))
print(len(publisher_values))

platform_nums = list(range(1, len(platform_values) + 1))
genre_nums = list(range(1, len(genre_values) + 1))
publisher_nums = list(range(1, len(publisher_values) + 1))

Random(0).shuffle(platform_nums)
Random(0).shuffle(genre_nums)
Random(0).shuffle(publisher_nums)

platform_dict = dict(zip(platform_values, platform_nums))
genre_dict = dict(zip(genre_values, genre_nums))
publisher_dict = dict(zip(publisher_values, publisher_nums))

df_work["Platform"] = df_work["Platform"].map(platform_dict)
df_work["Genre"] = df_work["Genre"].map(genre_dict)
df_work["Publisher"] = df_work["Publisher"].map(publisher_dict)

print(df_work.head())

df_work.drop(["Name"], axis=1, inplace=True)

print(df_work.head())

print(df_work.describe())

print(list(platform_dict.keys())[list(platform_dict.values()).index(9)])

print(list(publisher_dict.keys())[list(publisher_dict.values()).index(3)])

print(list(publisher_dict.keys())[list(publisher_dict.values()).index(3)])

df_work.corr()
plt.show()

upper_matrix = np.triu(df_work.corr())
sns.heatmap(df_work.corr(), annot=True, mask=upper_matrix)
plt.show()

fig = plt.figure(figsize=(14, 7))
sns.histplot(df["Platform"], kde=True)
plt.show()

fig = plt.figure(figsize=(14, 7))
sns.histplot(df["Genre"], kde=True)
plt.show()

fig = plt.figure(figsize=(14, 7))
sns.histplot(df_work["Publisher"], kde=True)
plt.show()

def publisher_by_index(i):
    return list(publisher_dict.keys())[list(publisher_dict.values()).index(i)]

pbhr_local_maxs = [
    [publisher_by_index(i) for i in range(40, 61)],
    [publisher_by_index(i) for i in range(180, 201)],
    [publisher_by_index(i) for i in range(270, 291)],
    [publisher_by_index(i) for i in range(340, 381)],
    [publisher_by_index(i) for i in range(470, 491)],
    [publisher_by_index(i) for i in range(540, 560)],
]
pbhr_local_mins = [
    [publisher_by_index(i) for i in range(62, 180)],
    [publisher_by_index(i) for i in range(202, 270)],
    [publisher_by_index(i) for i in range(292, 340)],
    [publisher_by_index(i) for i in range(382, 470)],
    [publisher_by_index(i) for i in range(492, 540)],
]

pbhr_global_maxs = [publisher_by_index(i) for i in range(465, 485)]
pbhr_global_mins = [
    [publisher_by_index(i) for i in range(99, 116)],
    [publisher_by_index(i) for i in range(450, 466)],
    [publisher_by_index(i) for i in range(500, 515)],
]

pbhr_other_values = [
    [publisher_by_index(i) for i in range(1, 40)],
    [publisher_by_index(i) for i in range(562, 579)],
]

for i in df_work.columns:
    sns.boxplot(df_work[i])
    plt.ylabel(f"{i}")
    plt.title("Boxplot диаграмма распределения")
    plt.show()

print(df[df.Year <= 1993].head())

sns.histplot(df_work["NA_Sales"])
plt.show()

log_NA_Sales = np.log1p(df_work["NA_Sales"])

sns.histplot(log_NA_Sales)
plt.show()

Q1 = log_NA_Sales.quantile(0.25)
Q3 = log_NA_Sales.quantile(0.75)
IQR = Q3 - Q1

no_otls_log_NA_Sales = log_NA_Sales[
    (log_NA_Sales >= Q1 - 1.5 * IQR) & (log_NA_Sales <= Q3 + 1.5 * IQR)
]

print(len(no_otls_log_NA_Sales))

sns.histplot(no_otls_log_NA_Sales)
plt.show()

sns.boxplot(pd.Series(no_otls_log_NA_Sales.reset_index(drop=True)))
plt.title("Boxplot диаграмма распределения")
plt.show()

print(len(df.NA_Sales[df.NA_Sales == 0]))

print(len(df_work.NA_Sales[df_work.NA_Sales == 0]))

print(len(df_work.EU_Sales[df_work.EU_Sales == 0]))

print(len(df_work.JP_Sales[df_work.JP_Sales == 0]))

print(len(df_work.Other_Sales[df_work.Other_Sales == 0]))

def add_sales_bin_col(df_):
    df_bin_sales_cols = df_.copy()
    region_sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

    for r in region_sales_cols:
        cur_col = "Are_" + r
        df_bin_sales_cols[cur_col] = df_bin_sales_cols[r].apply(lambda x: x > 0)

    return df_bin_sales_cols

df_work = add_sales_bin_col(df_work)

print(df_work.head())

print(["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])

df_work_NA = df_work[df_work.Are_NA_Sales == True]
df_work_EU = df_work[df_work.Are_EU_Sales == True]
df_work_JP = df_work[df_work.Are_JP_Sales == True]
df_work_Other = df_work[df_work.Are_Other_Sales == True]

print(df_work_EU.head())

sns.histplot(df_work["Global_Sales"])
plt.show()

log_Global_Sales = np.log1p(df_work["Global_Sales"])

sns.histplot(log_Global_Sales)
plt.show()

def del_outliers(column_):
    Q1 = column_.quantile(0.25)
    Q3 = column_.quantile(0.75)
    IQR = Q3 - Q1

    no_otls_col = column_[(column_ >= Q1 - 1.5 * IQR) & (column_ <= Q3 + 1.5 * IQR)]
    return no_otls_col

no_otls_log_Global_Sales = del_outliers(log_Global_Sales)

print(len(no_otls_log_Global_Sales))

sns.histplot(no_otls_log_Global_Sales)
plt.plot()

sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

for col in sales_cols:
    df_work[col] = np.log1p(df_work[col])

def get_lines_to_del(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column >= lower_bound) & (column <= upper_bound)

clear_sales = df_work[sales_cols].apply(get_lines_to_del)

print(clear_sales.isna().sum())

common_idxs = clear_sales.all(axis=1)

df_work_clear = df_work[common_idxs]

print(len(df_work_clear))

df_work_clear.Year = 2023 - df_work_clear.Year

print(df_work_clear.head())

upper_matrix = np.triu(df_work_clear.corr())
sns.heatmap(df_work_clear.corr(), annot=True, mask=upper_matrix)
plt.plot()

print(df_work_clear.corr()["Global_Sales"].drop(["Global_Sales"], axis=0).sort_values())

leak_columns = [
    "Rank",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Are_NA_Sales",
    "Are_EU_Sales",
    "Are_JP_Sales",
    "Are_Other_Sales",
]

df_final = df_work_clear.drop(leak_columns, axis=1)

print(df_final.head())

X = df_final.drop(["Global_Sales"], axis=1)
y = df_final["Global_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

print(X_train.shape)

print(X_test.shape)

models = [
    RandomForestRegressor(n_estimators=200, random_state=0),
    Lasso(alpha=0.01),
    ElasticNet(alpha=0.01, l1_ratio=0.5),
]

for model in models:
    print("Алгоритм: ", model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Ошибка: ", mean_absolute_error(y_pred, y_test))

params_grid = {
    "n_estimators": [50, 100, 200, 250, 300],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 3, 4, 5, 10, 15],
}

model = RandomForestRegressor(random_state=0)

gs = GridSearchCV(
    estimator=model,
    param_grid=params_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
gs.fit(X_train, y_train)

print(gs.best_params_)

model = RandomForestRegressor(
    max_depth=20, min_samples_split=15, n_estimators=300, random_state=0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Ошибка: ", mean_absolute_error(y_pred, y_test))

mae = mean_absolute_error(y_pred, y_test)
print(np.expm1(mae))

mean = df_final.Global_Sales.mean()
print(np.expm1(mean))

print(np.expm1(mae) / np.expm1(mean))
