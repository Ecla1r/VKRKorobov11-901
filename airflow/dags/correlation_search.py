import collections
import datetime
import math

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import telebot as tb
from scipy import stats
from itertools import combinations

DB_USER='postgres'
DB_PASS='postgres'
DB_HOST='db'
DB_PORT=5433
DB_NAME='correlation_test'

TG_BOT_TOKEN = '6268780388:AAHBuiltSKV_QD-hHYcyg0jIrQJWmVy6Yks'
TG_USER_ID = 419243340
bot = tb.TeleBot(TG_BOT_TOKEN)

a_level = 0.05
p_level = 1 - a_level

def connect():
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    conn = engine.connect()
    return conn


def check_data_type(data, column):
    if data[column].nunique() == 2:
        typo = 4
    elif 2 < data[column].nunique() <= 11:
        if data[column].dtype == 'object':
            typo = 2
        elif data[column].dtype == 'int64' or data[column].dtype == 'float64':
            typo = 3
        else:
            typo = 99
    elif data[column].nunique() > 11:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            typo = 1
        elif data[column].dtype == 'object':
            typo = 2
        else:
            typo = 99
    else:
        typo = 99

    return typo


def count_num_num(dataf, num1, num2):
    print(f'Checking correlation between {num1}(NUM) AND {num2}(NUM):')

    res = correlations(dataf, num1, num2)
    print(res.iloc[0])
    return res


def count_num_cat(dataf, num, cat):
    print(f'Checking correlation between {num}(NUM) AND {cat}(CAT):')

    numerical = dataf[num]
    categorical = dataf[cat]

    values = np.array(numerical)
    ss_total = np.sum((values.mean() - values) ** 2)

    cats = np.unique(categorical, return_inverse=True)[1]

    ss_betweengroups = 0

    for c in np.unique(cats):
        group = values[np.argwhere(cats == c).flatten()]
        ss_betweengroups += len(group) * (group.mean() - values.mean()) ** 2

    return np.sqrt(ss_betweengroups / ss_total)


def count_num_di(dataf, num, di):
    print(f'Checking correlation between {num}(NUM) AND {di}(DICHOTOMOUS):')

    res = stats.pointbiserialr(dataf[num], dataf[di])[0]
    return res


def count_cat_cat(dataf, cat1, cat2):
    print(f'Checking correlation between {cat1}(CAT_D) AND {cat2}(CAT_D):')

    log_base = 2
    s_xy = conditional_entropy(dataf[cat1], dataf[cat2], log_base)
    s_x = entropy(dataf[cat1], log_base)
    u = (s_x - s_xy) / s_x

    return u


def conditional_entropy(x, y, log_base: float = 2):
    y_counts = collections.Counter(y)
    xy_counts = collections.Counter(list(zip(x, y)))
    total_counts = len(x)

    cond_entropy = 0.0

    for xy in xy_counts.keys():
        p_xy = xy_counts[xy] / total_counts
        p_y = y_counts[xy[1]] / total_counts
        cond_entropy += p_xy * math.log(p_y / p_xy, log_base)

    return cond_entropy


def entropy(x, log_base):
    x_counts = collections.Counter(x)
    total_counts = len(x)
    p_x = list(map(lambda n: n / total_counts, x_counts.values()))
    entropy = 0.0
    for p in p_x:
        entropy += -p * math.log(p, log_base)

    return entropy


def count_cat_di(dataf, cat, di):
    print(f'Checking correlation between {cat}(CAT_O) AND {di}(DICHOTOMOUS):')

    res = stats.pointbiserialr(dataf[cat], dataf[di])[0]
    return res


def count_catDef_di(dataf, cat, di):
    print(f'Checking correlation between {cat}(CAT_D) AND {di}(DICHOTOMOUS):')

    cat_data = pd.get_dummies(dataf[cat])
    res = stats.chi2_contingency(cat_data, dataf[di])[1]
    return res


def count_catO_catO(dataf, cat1, cat2):
    print(f'Checking correlation between {cat1}(CAT_O) AND {cat2}(CAT_O):')

    res = dataf[[cat1, cat2]].corr(method='kendall').iloc[0, 1].round(10)
    return res


def count_di_di(dataf, cat1, cat2):
    print(f'Checking correlation between {cat1}(DICHOTOMOUS) AND {cat2}(DICHOTOMOUS):')

    res = stats.chi2_contingency(pd.crosstab(dataf[cat1], dataf[cat2]))[1]
    return res


def detect_outliers(data, column):
    mean = data[column].mean()
    sigma = data[column].std()
    outliers = data[(data[column] < mean - 3 * sigma) | (data[column] > mean + 3 * sigma)].index
    return outliers


def drop_anomalies(data):
    start = len(data)
    data = data.dropna(axis=0)
    print(f'deleted {start - len(data)} values')
    for col in data.columns:
        print(col)
        if check_data_type(data, col) == 1:
            out = set(detect_outliers(data, col))
            data.drop(out, inplace=True, axis=0)
    end = len(data)
    return data, start - end


def Shapiro_Wilk(data):
    data = np.array(data)
    result = stats.shapiro(data)
    a_calc = result.pvalue
    if a_calc >= a_level:
        norm = True
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, 10)} >= a_level = {round(a_level, 10)}" + \
                              ", то ПРИНИМАЕТСЯ гипотеза о нормальности распределения по критерию Шапиро-Уилка"
    else:
        norm = False
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, 10)} < a_level = {round(a_level, 10)}" + \
                              ", то ОТВЕРГАЕТСЯ гипотеза о нормальности распределения по критерию Шапиро-Уилка"
    print(conclusion_ShW_test)
    return norm


def interval_counter(x):
    pred = round(3.31 * math.log(x, 10) + 1)
    if pred >= 2:
        return pred
    else:
        return 2


def chaddock_scale_check(r, name='r'):
    chaddock_scale = {
        f'отсутствует (|{name}| <= 0.1)': 0.1,
        f'очень слабая (0.1 < |{name}| <= 0.2)': 0.2,
        f'слабая (0.2 < |{name}| <= 0.3)': 0.3,
        f'умеренная (0.3 < |{name}| <= 0.5)': 0.5,
        f'ощутимая (0.5 < |{name}| <= 0.7)': 0.7,
        f'высокая (0.7 < |{name}| <= 0.9)': 0.9,
        f'очень высокая (0.9 < |{name}| <= 0.99)': 0.99,
        f'функциональная (|{name}| > 0.99)': 1.0}

    r_scale = list(chaddock_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_chaddock_scale = list(chaddock_scale.keys())[i]
            break
    return conclusion_chaddock_scale


def Evans_scale_check(r, name='r'):
    Evans_scale = {
        f'очень слабая (|{name}| < 0.19)': 0.2,
        f'слабая (0.2 < |{name}| <= 0.39)': 0.4,
        f'умеренная (0.4 < |{name}| <= 0.59)': 0.6,
        f'сильная (0.6 < |{name}| <= 0.79)': 0.8,
        f'очень сильная (0.8 < |{name}| <= 1.0)': 1.0}

    r_scale = list(Evans_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Evans_scale = list(Evans_scale.keys())[i]
            break
    return conclusion_Evans_scale


def correlations(df, col1, col2):
    X = np.array(df[col1])
    Y = np.array(df[col2])

    XY_df = pd.DataFrame({'X': X, 'Y': Y})

    if 100 < len(X) < 1200:
        norm1 = Shapiro_Wilk(X)
        norm2 = Shapiro_Wilk(Y)
        if not (norm1 & norm2):
            print('DISTRIBUTION IS NOT NORMAL')

    matrix_XY_df = XY_df.copy()

    n_X = len(X)
    n_Y = len(Y)

    K_X = interval_counter(n_X)
    K_Y = interval_counter(n_Y)

    cut_X = pd.cut(X, bins=K_X)
    cut_Y = pd.cut(Y, bins=K_Y)

    matrix_XY_df['cut_X'] = cut_X
    matrix_XY_df['cut_Y'] = cut_Y

    CorrTable_df = pd.crosstab(
        index=matrix_XY_df['cut_X'],
        columns=matrix_XY_df['cut_Y'],
        rownames=['cut_X'],
        colnames=['cut_Y'])

    CorrTable_np = np.array(CorrTable_df)

    n_group_X = [np.sum(CorrTable_np[i]) for i in range(K_X)]
    n_group_Y = [np.sum(CorrTable_np[:, j]) for j in range(K_Y)]

    Xboun_mean = [(CorrTable_df.index[i].left + CorrTable_df.index[i].right) / 2 for i in range(K_X)]
    Xboun_mean[0] = (np.min(X) + CorrTable_df.index[0].right) / 2
    Xboun_mean[K_X - 1] = (CorrTable_df.index[K_X - 1].left + np.max(X)) / 2

    Yboun_mean = [(CorrTable_df.columns[j].left + CorrTable_df.columns[j].right) / 2 for j in range(K_Y)]
    Yboun_mean[0] = (np.min(Y) + CorrTable_df.columns[0].right) / 2
    Yboun_mean[K_Y - 1] = (CorrTable_df.columns[K_Y - 1].left + np.max(Y)) / 2

    Xmean_group = [np.sum(CorrTable_np[:, j] * Xboun_mean) / n_group_Y[j] for j in range(K_Y)]
    Ymean_group = [np.sum(CorrTable_np[i] * Yboun_mean) / n_group_X[i] for i in range(K_X)]

    disp_total_X = np.sum(n_group_X * (Xboun_mean - np.mean(X)) ** 2)
    disp_total_Y = np.sum(n_group_Y * (Yboun_mean - np.mean(Y)) ** 2)

    disp_between_X = np.sum(n_group_Y * (Xmean_group - np.mean(X)) ** 2)
    disp_between_Y = np.sum(n_group_X * (Ymean_group - np.mean(Y)) ** 2)

    corr_ratio_XY = math.sqrt(disp_between_Y / disp_total_Y)
    return_corr = corr_ratio_XY

    return_corr_evans = Evans_scale_check(corr_ratio_XY, name=chr(951))
    return_corr_chaddock = chaddock_scale_check(corr_ratio_XY, name=chr(951))

    F_corr_ratio_calc = (n_X - K_X) / (K_X - 1) * corr_ratio_XY ** 2 / (1 - corr_ratio_XY ** 2)
    dfn = K_X - 1
    dfd = n_X - K_X
    F_corr_ratio_table = stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
    if F_corr_ratio_calc < F_corr_ratio_table:
        return_corr_meaning = False
        conclusion_corr_ratio_sign = f"Так как F_calc < F_table" + \
                                     ", то гипотеза о равенстве нулю корреляционного отношения ПРИНИМАЕТСЯ, т.е. корреляционная связь НЕЗНАЧИМА"
    else:
        return_corr_meaning = True
        conclusion_corr_ratio_sign = f"Так как F_calc >= F_table" + \
                                     ", то гипотеза о равенстве нулю корреляционного отношения ОТВЕРГАЕТСЯ, т.е. корреляционная связь ЗНАЧИМА"

    if return_corr_meaning:

        corr_coef = stats.pearsonr(X, Y)[0]

        a_corr_coef_calc = stats.pearsonr(X, Y)[1]
        if a_corr_coef_calc >= a_level:
            return_lin_meaning = False
            conclusion_corr_coef_sign = f"Так как a_calc >= a_level" + \
                                        ", то гипотеза о равенстве нулю коэффициента линейной корреляции ПРИНИМАЕТСЯ, т.е. линейная корреляционная связь НЕЗНАЧИМА"
        else:
            return_lin_meaning = True
            conclusion_corr_coef_sign = f"Так как a_calc < a_level" + \
                                        ", то гипотеза о равенстве нулю коэффициента линейной корреляции ОТВЕРГАЕТСЯ, т.е. линейная корреляционная связь ЗНАЧИМА"
        F_line_corr_sign_calc = (n_X - K_X) / (K_X - 2) * (corr_ratio_XY ** 2 - corr_coef ** 2) / (
                1 - corr_ratio_XY ** 2)

        dfn = K_X - 2
        dfd = n_X - K_X
        F_line_corr_sign_table = stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
        if F_line_corr_sign_calc < F_line_corr_sign_table:
            return_linear = True
            conclusion_line_corr_sign = f"Так как F_calc < F_table =" + \
                                        f", то гипотеза о равенстве {chr(951)} и r ПРИНИМАЕТСЯ, т.е. корреляционная связь ЛИНЕЙНАЯ"
        else:
            return_linear = False
            conclusion_line_corr_sign = f"Так как F_calc >= F_table" + \
                                        f", то гипотеза о равенстве {chr(951)} и r ОТВЕРГАЕТСЯ, т.е. корреляционная связь НЕЛИНЕЙНАЯ"

        return pd.DataFrame({'corr': [return_corr],
                             'evans': [return_corr_evans],
                             'chaddock': [return_corr_chaddock],
                             'meaning': [return_corr_meaning],
                             'lin_meaning': [return_lin_meaning],
                             'lin': [return_linear]})
    else:
        return pd.DataFrame({'corr': [return_corr],
                             'meaning': [return_corr_meaning]})


def main():
    con = connect()
    res_sql = con.execute(text(
        'SELECT table_name FROM information_schema.tables WHERE table_schema=\'public\' AND table_type=\'BASE TABLE\';'))
    tables = res_sql.fetchall()
    for table in tables:
        result = ''
        res_tg = list()
        table_name = table[0]
        data = pd.read_sql("SELECT * FROM " + table_name, con)

        df = data
        print(len(df))

        df, deleted = drop_anomalies(df)
        print(f'Deleted {deleted} values')

        for col in df.columns:
            if df[col].nunique() == 2 and df[col].dtype == 'object':
                uniques = df[col].unique()
                df[col] = df[col].replace([uniques[0], uniques[1]], [0, 1])

        cols = list(combinations(df.columns, 2))
        i = 0
        for pair in cols:
            i += 1
            print(f'Iter {i}/{len(cols)}')
            # typo == 1: NUMERICAL
            # typo == 2: CATEGORICAL NOMINAL
            # typo == 3: CATEGORICAL ORDINAL
            # typo == 4: CATEGORICAL DICHOTOMOUS (2 Values)
            typo1 = check_data_type(df, pair[0])
            typo2 = check_data_type(df, pair[1])
            print(f'Types {typo1} and {typo2}')

            if typo1 == 1 and typo2 == 1:
                res = count_num_num(df, pair[0], pair[1])
                corr_meaning = res['meaning'].iat[0]
                corr = res['corr'].iat[0]
                if corr_meaning:
                    ev_close = res['evans'].iat[0]
                    ch_close = res['chaddock'].iat[0]
                    if res['lin_meaning'].iat[0]:
                        corr_lin = res['lin'].iat[0]
                        hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Количественный) ' \
                              f'имеют линейную корреляцию = {corr}.\n' \
                              f'Корреляция {ev_close} по шкале Эванса и {ch_close} по шкале Чеддока\n\n'
                        result += hyp
                        if abs(corr) >= 0.35:
                            res_tg.append(f'[{table_name}] {hyp}')
                    else:
                        hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Количественный) ' \
                              f'имеют нелинейную корреляцию = {corr}.\n' \
                              f'Корреляция {ev_close} по шкале Эванса и {ch_close} по шкале Чеддока\n\n'
                        result += hyp
                        if abs(corr) >= 0.35:
                            res_tg.append(f'[{table_name}] {hyp}')
                else:
                    hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Количественный) ' \
                          f'не имеют корреляции\n\n'
                    result += hyp
            if typo1 == 1 and typo2 == 2:
                res = count_num_cat(df, pair[0], pair[1])
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Категориальный Номинальный) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 1 and typo2 == 3:
                res = (count_num_cat(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Категориальный Порядковый) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 1 and typo2 == 4:
                res = (count_num_di(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Количественный) и {pair[1]} (Тип: Дихотомический) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 2 and typo2 == 1:
                res = (count_num_cat(df, pair[1], pair[0]))
                hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Количественный) и {pair[0]} (Тип: Категориальный Номинальный) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 2 and typo2 == 2:
                res = (count_cat_cat(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Номинальный) и {pair[1]} (Тип: Категориальный Номинальный) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 2 and typo2 == 3:
                res = (count_cat_cat(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Номинальный) и {pair[1]} (Тип: Категориальный Порядковый) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 2 and typo2 == 4:
                res = (count_catDef_di(df, pair[0], pair[1]))
                if res < a_level:
                    hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Номинальный) и {pair[1]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляция существует, так как p-значение={res} меньше допустимого\n\n'
                    result += hyp
                    res_tg.append(f'[{table_name}] {hyp}')
                else:
                    hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Номинальный) и {pair[1]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляции нет, так как p-значение={res} выше допустимого\n\n'
                    result += hyp

            if typo1 == 3 and typo2 == 1:
                res = (count_num_cat(df, pair[1], pair[0]))
                hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Количественный) и {pair[0]} (Тип: Категориальный Порядковый) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 3 and typo2 == 2:
                res = (count_cat_cat(df, pair[1], pair[0]))
                hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Категориальный Номинальный) и {pair[0]} (Тип: Категориальный Порядковый) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 3 and typo2 == 3:
                res = (count_catO_catO(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Порядковый) и {pair[1]} (Тип: Категориальный Порядковый) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 3 and typo2 == 4:
                res = (count_cat_di(df, pair[0], pair[1]))
                hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Категориальный Порядковый) и {pair[1]} (Тип: Дихотомический) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 4 and typo2 == 1:
                res = (count_num_di(df, pair[1], pair[0]))
                hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Количественный) и {pair[0]} (Тип: Дихотомический) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 4 and typo2 == 2:
                res = (count_catDef_di(df, pair[1], pair[0]))
                if res < a_level:
                    hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Категориальный Номинальный) и {pair[0]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляция существует, так как p-значение={res} меньше допустимого\n\n'
                    result += hyp
                    res_tg.append(f'[{table_name}] {hyp}')
                else:
                    hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Категориальный Номинальный) и {pair[0]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляции нет, так как p-значение={res} выше допустимого\n\n'
                    result += hyp
            if typo1 == 4 and typo2 == 3:
                res = (count_cat_di(df, pair[1], pair[0]))
                hyp = f'Гипотеза: Признаки {pair[1]} (Тип: Категориальный Порядковый) и {pair[0]} (Тип: Дихотомический) ' \
                      f'имеют корреляцию = {res}\n\n'
                result += hyp
                if abs(res) >= 0.35:
                    res_tg.append(f'[{table_name}] {hyp}')
            if typo1 == 4 and typo2 == 4:
                res = (count_di_di(df, pair[0], pair[1]))
                if res < a_level:
                    hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Дихотомический) и {pair[1]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляция существует, так как p-значение={res} меньше допустимого\n\n'
                    result += hyp
                    res_tg.append(f'[{table_name}] {hyp}')
                else:
                    hyp = f'Гипотеза: Признаки {pair[0]} (Тип: Дихотомический) и {pair[1]} (Тип: Дихотомический) ' \
                          f'коррелируют следующим образом: Корреляции нет, так как p-значение={res} выше допустимого\n\n'
                    result += hyp
            if typo1 == 99:
                res = f'Ошибка при проверки корреляций между признаками {pair[0]} и {pair[1]} - неизвестный тип данных для признака {pair[0]}!\n\n'
                result += res
            if typo2 == 99:
                res = f'Ошибка при проверки корреляций между признаками {pair[0]} and {pair[1]} - неизвестный тип данных для признака {pair[1]}!\n\n'
                result += res
            print('\n')
        result += f'Processed {i} pairs of columns.'
        curDate = datetime.datetime.now()
        curDate = str(curDate).replace(' ', '_').replace(':', '_')
        curDate = curDate[:-7]
        path_file = f'output/{table_name} correlations {curDate}.txt'
        print(path_file)
        with open(path_file, 'w', encoding="utf-8") as f:
            f.write(result)

        for corr in res_tg:
            bot.send_message(TG_USER_ID, corr)

    con.close()


if __name__ == '__main__':
    main()
