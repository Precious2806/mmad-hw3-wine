import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent
st.set_page_config(page_title='Оценка вина', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def cached_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
p_test = cached_proba(model, X_test)

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}

st.title('Какие вина включить в подборку')
threshold = st.sidebar.slider(
    'Порог включения в подборку', 0.0, 1.0,
    value=float(config['threshold']), step=0.001, format='%.3f', key='threshold')
st.sidebar.caption('Выше порог - строже отбор, ниже порог - шире подборка')

one_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Загрузить CSV', 'Что выучила модель'
])

with one_tab:
    values = {}
    columns = st.columns(3)
    numeric_features = [name for name in features if name != 'is_red']

    for i, name in enumerate(numeric_features):
        step = 0.00001 if name == 'density' else 0.001 if name == 'chlorides' else 0.01
        with columns[i % 3]:
            values[name] = st.number_input(
                labels[name], min_value=float(X_test[name].min()),
                max_value=float(X_test[name].max()),
                value=float(X_test[name].median()), step=step,
                format='%.5f' if name == 'density' else '%.3f', key=name)

    values['is_red'] = int(st.toggle('Красное вино', value=False))
    row = pd.DataFrame([values], columns=features)
    probability = float(cached_proba(model, row)[0])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Включить в подборку')
    else:
        st.info('Оставить вне подборки')

with metrics_tab:
    prediction = (p_test >= threshold).astype(int)
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric('Precision', f'{precision:.3f}')
    col2.metric('Recall', f'{recall:.3f}')
    col3.metric('F1', f'{f1:.3f}')

    cm = metrics.confusion_matrix(y_test, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr, re, _ = metrics.precision_recall_curve(y_test, p_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), recall, color='red', label='Текущий порог')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    axes[0].legend()
    axes[1].plot(re, pr)
    axes[1].axhline(y_test.mean(), linestyle='--', color='gray')
    axes[1].scatter(recall, precision, color='red', label='Текущий порог')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('При повышении порога в подборку обычно попадает меньше вин, зато отбор становится строже. '
             'При снижении порога находим больше хороших вин, но чаще включаем посредственные.')

with batch_tab:
    st.write('Загрузи CSV с характеристиками вин, порог используется тот же, что и слева')
    st.caption('Нужны столбцы: ' + ', '.join(features))
    uploaded = st.file_uploader('CSV с винами', type=['csv'])

    if uploaded is not None:
        try:
            uploaded.seek(0)
            batch = pd.read_csv(uploaded)
            missing = [name for name in features if name not in batch.columns]

            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            elif batch.empty:
                st.error('В файле нет строк с винами')
            else:
                X_batch = batch[features].apply(pd.to_numeric, errors='coerce')
                if not np.isfinite(X_batch.to_numpy()).all():
                    st.error('В признаках есть пропуски или значения, которые не являются числами')
                elif not X_batch['is_red'].isin([0, 1]).all():
                    st.error('В столбце is_red должны быть только 0 или 1')
                else:
                    result = batch.copy()
                    probabilities = cached_proba(model, X_batch)
                    result['probability'] = probabilities
                    result['prediction'] = (probabilities >= threshold).astype(int)
                    st.dataframe(result)
                    st.download_button(
                        'Скачать результат', result.to_csv(index=False).encode('utf-8-sig'),
                        file_name='wine_predictions.csv', mime='text/csv')
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
            st.error('Не получилось прочитать CSV, проверь разделители и числовые значения')

with weights_tab:
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue' if value < 0 else 'seagreen' for value in coefficients]
    ax.barh([labels[name] for name in coefficients.index], coefficients.values, color=colors)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + labels[coefficients.idxmax()])
    st.write('Сильнее всего против: ' + labels[coefficients.idxmin()])
    st.caption('Это связи, найденные моделью при учёте остальных признаков, они не доказывают причинное влияние')
