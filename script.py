import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# вынос констант
Z_THRESHOLD = 3.5
MIN_OTS_MULTIPLIER = 4.0
MIN_ABS_OTS = 5.0
MIN_GROUP_SIZE = 10


# загрузка данных из всех parquet файлов в директории
def load_and_prepare_data(directory='.'):
    path = Path(directory)
    file_list = [str(p) for p in path.rglob('part-*.parquet')]

    print(f'найдено {len(file_list)} файлов')
    
    df_list = [pd.read_parquet(f) for f in file_list]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f'успешно объединено. итоговый размер: {df.shape[0]} строк, {df.shape[1]} колонок')
    print('превью первых 3 строк загруженных данных:')
    print(df.head(3).to_string())

    df = df[df['BrandinDelivery'] == 1].copy()
    df = df[df['CategoryNameDelivery'].notna() & (df['CategoryNameDelivery'] != '')].copy()

    # приведение типов
    df['researchdate'] = pd.to_datetime(df['researchdate']).dt.date
    df['Weight'] = df['Weight'].astype(float)

    return df


# подсчет ots
def calculate_daily_ots(df):
    counts = df.groupby(['SubjectID', 'researchdate', 'CategoryNameDelivery', 'BrandID', 'Brand'])['Weight'].agg(
        count_rows='count',
        mean_weight='mean'
    ).reset_index()
    
    counts['daily_ots'] = counts['mean_weight'] * counts['count_rows']
    
    available_demo = [c for c in ['Пол', 'Возраст', 'Регион', 'Федеральный_округ', 'ResourceName', 'ResourceType', 'Platform', 'UseType'] if c in df.columns]
    
    if available_demo:
        demo = df.groupby(['SubjectID', 'researchdate'])[available_demo].first().reset_index()
        counts = counts.merge(demo, on=['SubjectID', 'researchdate'], how='left')
        
    return counts, df


# поиск аномалий
def detect_anomalies(counts_df):
    def calc_robust_z(group):
        median = np.median(group['daily_ots'])
        mad = np.median(np.abs(group['daily_ots'] - median))
        
        # чтобы не делить на 0
        epsilon = 1e-6
        if mad < epsilon:
            mad = np.std(group['daily_ots']) + epsilon
            
        group['median_ref'] = median
        group['mad_ref'] = mad
        group['score'] = (group['daily_ots'] - median) / (1.4826 * mad)

        return group

    g1 = counts_df.groupby(['researchdate', 'CategoryNameDelivery']).filter(lambda x: len(x) >= MIN_GROUP_SIZE)
    g1 = g1.groupby(['researchdate', 'CategoryNameDelivery']).apply(calc_robust_z).reset_index(drop=True)
    
    g2 = counts_df[~counts_df.index.isin(g1.index)]
    g2 = g2.groupby(['CategoryNameDelivery']).apply(calc_robust_z).reset_index(drop=True)
    
    scored_df = pd.concat([g1, g2], ignore_index=True)
    
    # защита от отсеивания малых значений
    dynamic_threshold = scored_df['median_ref'] * MIN_OTS_MULTIPLIER + MIN_ABS_OTS
    
    scored_df['is_anomaly'] = (scored_df['score'] > Z_THRESHOLD) & (scored_df['daily_ots'] > MIN_ABS_OTS) & (scored_df['daily_ots'] > dynamic_threshold)
    
    return scored_df


# выходные файлы
def generate_outputs(scored_df, original_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # anomaly_reasons.csv
    anomalies_detailed = scored_df[scored_df['is_anomaly']].copy()
    anomalies_detailed['threshold'] = anomalies_detailed['median_ref'] * MIN_OTS_MULTIPLIER + MIN_ABS_OTS
    anomalies_detailed['reason'] = anomalies_detailed['threshold'].apply(lambda x: f'z-score > {Z_THRESHOLD} & ots > порога ({x:.2f})')
    
    cols_to_save = ['SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryNameDelivery', 'daily_ots', 'score', 'threshold', 'reason']
    available_cols = [c for c in cols_to_save if c in anomalies_detailed.columns]
    anomalies_detailed[available_cols].to_csv(os.path.join(OUTPUT_DIR, 'anomaly_reasons.csv'), index=False)
    
    # anomalies.csv 
    removal_pairs = anomalies_detailed[['SubjectID', 'researchdate']].drop_duplicates()
    removal_pairs.to_csv(os.path.join(OUTPUT_DIR, 'anomalies.csv'), index=False)
    
    # метрики для графиков
    original_df['researchdate'] = pd.to_datetime(original_df['researchdate']).dt.date
    merged = original_df.merge(removal_pairs, on=['SubjectID', 'researchdate'], how='left', indicator=True)
    merged['is_removed'] = merged['_merge'] == 'both'
    
    # фильтрация строк участвовавших в анализе
    analysis_mask = (original_df['BrandinDelivery'] == 1) & (original_df['CategoryNameDelivery'].notna())
    df_analysis = original_df[analysis_mask].copy()
    df_analysis = df_analysis.merge(removal_pairs, on=['SubjectID', 'researchdate'], how='left', indicator=True)
    df_analysis['is_removed'] = df_analysis['_merge'] == 'both'
    
    df_analysis['ots'] = df_analysis['Weight'] 
    
    return df_analysis, removal_pairs


# графики
def plot_results(df_analysis):
    sns.set_theme(style='whitegrid')
    
    # график 1: total_ots_before_after.png
    plt.figure(figsize=(12, 6))
    daily_ots = df_analysis.groupby(['researchdate', 'is_removed'])['ots'].sum().unstack().fillna(0)
    daily_ots['Total'] = daily_ots[True] + daily_ots[False]
    
    daily_ots[False].plot(kind='bar', stacked=True, label='оставшийся ots', color='skyblue', ax=plt.gca())
    daily_ots[True].plot(kind='bar', stacked=True, label='удаленный ots (аномалии)', color='salmon', ax=plt.gca())
    
    plt.title('изменение общего ots до и после удаления аномалий по дням')
    plt.xlabel('дата')
    plt.ylabel('суммарный ots')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'total_ots_before_after.png'), dpi=300)
    plt.close()
    
    # график 2: category_ots_change.png
    plt.figure(figsize=(10, 6))
    cat_ots = df_analysis.groupby(['CategoryNameDelivery', 'is_removed'])['ots'].sum().unstack().fillna(0)
    cat_ots['Total'] = cat_ots[True] + cat_ots[False]
    cat_ots['Change_%'] = (cat_ots[True] / cat_ots['Total']) * 100
    
    cat_ots['Change_%'].sort_values(ascending=False).plot(kind='bar', color='teal', ax=plt.gca())
    plt.title('изменение ots по categorydelivery в процентах (доля удаленного)')
    plt.xlabel('categorydelivery')
    plt.ylabel('процент удаления (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'category_ots_change.png'), dpi=300)
    plt.close()
    
    # график 3: daily_anomaly_count.png
    plt.figure(figsize=(12, 6))
    daily_counts = df_analysis[df_analysis['is_removed']].groupby('researchdate')['SubjectID'].nunique()
    daily_counts.plot(kind='bar', color='coral', ax=plt.gca())
    plt.title('количество аномальных респондентов по дням')
    plt.xlabel('дата')
    plt.ylabel('количество уникальных subjectid')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'daily_anomaly_count.png'), dpi=300)
    plt.close()


# аналитические возможности
def analyze_demographics(df_analysis):
    # графики до/после по характеристикам респондентов
    demo_cols = [c for c in ['Пол', 'Возраст', 'Регион', 'Федеральный_округ'] if c in df_analysis.columns]
    for col in demo_cols:
        plt.figure(figsize=(10, 5))
        cross = df_analysis.groupby([col, 'is_removed'])['ots'].sum().unstack().fillna(0)
        cross['Total'] = cross[True] + cross[False]
        cross['Removed_%'] = (cross[True] / cross['Total']) * 100
        cross['Removed_%'].sort_values(ascending=False).plot(kind='bar', color='purple')
        plt.title(f'доля удаленного ots по {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'demo_{col}.png'), dpi=300)
        plt.close()


def analyze_resources(df_analysis):
    # графики до/после по характеристикам ресурсов
    res_cols = [c for c in ['ResourceName', 'ResourceType', 'Platform', 'UseType'] if c in df_analysis.columns]
    for col in res_cols:
        plt.figure(figsize=(10, 5))
        cross = df_analysis.groupby([col, 'is_removed'])['ots'].sum().unstack().fillna(0)
        cross['Total'] = cross[True] + cross[False]
        cross['Removed_%'] = (cross[True] / cross['Total']) * 100
        cross['Removed_%'].sort_values(ascending=False).plot(kind='bar', color='orange')
        plt.title(f'доля удаленного ots по {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'resource_{col}.png'), dpi=300)
        plt.close()


def get_anomaly_queries(original_df, subject_id, date):
    # таблица поисковых запросов для выбранного респондента в определенную дату
    date = pd.to_datetime(date).date()
    queries = original_df[(original_df['SubjectID'] == subject_id) & 
                          (pd.to_datetime(original_df['researchdate']).dt.date == date)]
    return queries[['researchdate', 'QueryText', 'Brand', 'CategoryNameDelivery', 'Weight']]


def plot_brand_ots_over_time(df_analysis, brand_id):
    # график изменения ots по дням для выбранного бренда до и после очистки
    df_brand = df_analysis[df_analysis['BrandID'] == brand_id].copy()
    daily = df_brand.groupby(['researchdate', 'is_removed'])['ots'].sum().unstack().fillna(0)
    
    plt.figure(figsize=(12, 6))
    daily[False].plot(kind='line', marker='o', label='оставшийся ots', color='blue', ax=plt.gca())
    daily[True].plot(kind='line', marker='x', label='удаленный ots', color='red', ax=plt.gca())
    
    plt.title(f'динамика ots для brandid={brand_id} до и после очистки')
    plt.xlabel('дата')
    plt.ylabel('ots')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'brand_{brand_id}_ots_over_time.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    print('запуск алгоритма')

    df = load_and_prepare_data('.')
    counts_df, original_df = calculate_daily_ots(df)
    scored_df = detect_anomalies(counts_df)
    df_analysis, removal_pairs = generate_outputs(scored_df, original_df)
    
    plot_results(df_analysis)
    
    print('генерация аналитических резервов')

    analyze_demographics(df_analysis)
    analyze_resources(df_analysis)

    print('Done !')
    
    total_respondents = df['SubjectID'].nunique()
    removed_respondents = removal_pairs['SubjectID'].nunique()
    print(f'всего респондентов в данных: {total_respondents}')
    print(f'удалено респондентов: {removed_respondents}')
    print(f'доля удаленных: {(removed_respondents/total_respondents)*100:.2f}%')
