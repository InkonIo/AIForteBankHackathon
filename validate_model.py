"""
🔍 ГЛУБОКАЯ ВАЛИДАЦИЯ ML-МОДЕЛИ
Проверка на переобучение, утечку данных и реальную производительность
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve)
import catboost as cb
import lightgbm as lgb
import psycopg2
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin

# ==================== ОБЁРТКА ДЛЯ LIGHTGBM ====================

class LGBMWrapper(BaseEstimator, ClassifierMixin):
    """Обёртка для LightGBM Booster для совместимости с sklearn"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.booster_ = None
        self.classes_ = np.array([0, 1])
        self.n_classes_ = 2
        
    def fit(self, X, y):
        """Загружаем pre-trained модель (не обучаем заново)"""
        if self.booster_ is None:
            self.booster_ = lgb.Booster(model_file=self.model_path)
        return self
    
    def predict_proba(self, X):
        """Предсказать вероятности"""
        if self.booster_ is None:
            self.booster_ = lgb.Booster(model_file=self.model_path)
        
        # Конвертируем DataFrame в numpy если нужно
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        proba = self.booster_.predict(X_array)
        return np.vstack([1 - proba, proba]).T
    
    def predict(self, X):
        """Предсказать классы"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_data_from_db():
    """Загрузить данные из PostgreSQL БД"""
    print("\n📥 Загрузка данных из БД...")
    
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fraud_db',
        'user': 'postgres',
        'password': 'Alikhancool20!'
    }
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        t.id,
        t.transaction_id,
        t.customer_id,
        t.recipient_id,
        t.amount,
        t.transaction_datetime,
        t.is_fraud,
        
        -- Временные признаки
        EXTRACT(HOUR FROM t.transaction_datetime) as hour,
        EXTRACT(MINUTE FROM t.transaction_datetime) as minute,
        EXTRACT(DOW FROM t.transaction_datetime) as day_of_week,
        EXTRACT(DAY FROM t.transaction_datetime) as day_of_month,
        EXTRACT(MONTH FROM t.transaction_datetime) as month,
        
        -- Поведенческие паттерны
        cb.avg_logins_per_day_30d,
        cb.avg_logins_per_day_7d,
        cb.avg_session_interval_sec,
        cb.burstiness_score,
        cb.exp_weighted_avg_interval,
        cb.fano_factor,
        cb.interval_zscore,
        cb.latest_os_version,
        cb.latest_phone_model,
        cb.login_freq_change_ratio,
        cb.login_ratio_7d_30d,
        cb.logins_last_30_days,
        cb.logins_last_7_days,
        cb.session_interval_std,
        cb.session_interval_variance,
        cb.unique_os_versions_30d,
        cb.unique_phone_models_30d
        
    FROM transactions t
    LEFT JOIN customer_behavior_patterns cb 
        ON t.customer_id = cb.customer_id 
        AND DATE(t.transaction_datetime) = cb.trans_date  
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Преобразование времени
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
    df['is_fraud'] = df['is_fraud'].astype(bool)
    
    print(f"✅ Загружено {len(df)} транзакций")
    print(f"   Мошеннических: {df['is_fraud'].sum()}")
    print(f"   Чистых: {(~df['is_fraud']).sum()}")
    
    return df

# ==================== FEATURE ENGINEERING ====================

def engineer_features_advanced(df):
    """Создать РАСШИРЕННЫЙ набор признаков (копия из train_improved_model.py)"""
    
    print("\n🔧 Feature Engineering...")
    
    df = df.copy()
    
    # ========== СНАЧАЛА ЗАПОЛНЯЕМ ПРОПУСКИ ==========
    df = df.fillna(0)
    
    # ========== ГРУППА 1: ПРИЗНАКИ СУММЫ ==========
    
    # Базовые трансформации
    df['amount_log'] = np.log(df['amount'] + 1)
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['amount_cbrt'] = np.cbrt(df['amount'])
    
    # Категории сумм (ИСПРАВЛЕНО!)
    df['amount_category'] = pd.cut(df['amount'], 
                                    bins=[0, 10000, 50000, 100000, 500000, float('inf')],
                                    labels=[0, 1, 2, 3, 4])
    df['amount_category'] = df['amount_category'].fillna(0).astype(int)
    
    # Признаки округлости
    df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
    df['is_round_1000'] = (df['amount'] % 1000 == 0).astype(int)
    df['is_round_10000'] = (df['amount'] % 10000 == 0).astype(int)
    
    # ========== ГРУППА 2: ВРЕМЕННЫЕ ПРИЗНАКИ ==========
    
    # Базовые флаги
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] < 6)).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 6) & (df['hour'] < 9)).astype(int)
    df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    
    # Опасные часы (пик мошенничества: 2-5 утра)
    df['is_peak_fraud_hour'] = ((df['hour'] >= 2) & (df['hour'] < 5)).astype(int)
    
    # Циклические признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ========== ГРУППА 3: ПОВЕДЕНЧЕСКИЕ ПРИЗНАКИ ==========
    
    # Флаг наличия данных
    df['has_behavior_data'] = ((df['logins_last_7_days'] > 0) | 
                                (df['logins_last_30_days'] > 0)).astype(int)
    
    # Активность клиента
    df['login_activity_score'] = df['logins_last_7_days'] / (df['logins_last_30_days'] + 1)
    
    # Аномалии в поведении
    df['is_zero_activity'] = ((df['logins_last_7_days'] == 0) & 
                               (df['logins_last_30_days'] == 0)).astype(int)
    
    df['is_high_burstiness'] = (df['burstiness_score'] > 0.7).astype(int)
    df['is_unusual_interval'] = (np.abs(df['interval_zscore']) > 2).astype(int)
    
    # Признаки устройств
    df['has_os_data'] = (df['latest_os_version'] != 0).astype(int)
    df['has_phone_data'] = (df['latest_phone_model'] != 0).astype(int)
    df['device_diversity'] = df['unique_os_versions_30d'] + df['unique_phone_models_30d']
    df['is_multi_device'] = (df['device_diversity'] > 2).astype(int)
    
    # ========== ГРУППА 4: КОМБИНИРОВАННЫЕ ПРИЗНАКИ ==========
    
    # Ночь + большая сумма
    df['night_large_amount'] = (df['is_night'] * (df['amount'] > 100000)).astype(int)
    
    # Выходной + ночь
    df['weekend_night'] = (df['is_weekend'] * df['is_night']).astype(int)
    
    # Большая сумма + нет активности
    df['large_amount_no_activity'] = ((df['amount'] > 100000) * 
                                       df['is_zero_activity']).astype(int)
    
    # Круглая сумма + ночь
    df['round_amount_night'] = (df['is_round_10000'] * df['is_night']).astype(int)
    
    # ========== ФИНАЛЬНАЯ ОЧИСТКА ==========
    
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Создано {len(df.columns) - 34} признаков")
    
    return df

# ==================== ФУНКЦИИ ВАЛИДАЦИИ ====================

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Построить матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean', 'Fraud'],
                yticklabels=['Clean', 'Fraud'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Добавляем метрики
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    plt.text(2.5, 0.5, metrics_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_proba):
    """Построить ROC кривую"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_proba):
    """Построить Precision-Recall кривую"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_learning_curves(estimator, X, y, cv=5):
    """Построить кривые обучения"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.1, color='g')
    
    plt.xlabel('Training Examples')
    plt.ylabel('AUC Score')
    plt.title('Learning Curves', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Добавляем информацию о переобучении
    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.1:
        plt.text(0.5, 0.5, f'⚠️ Overfitting detected!\nGap: {gap:.3f}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()

def cross_validate_model(model, X, y, cv=5):
    """Кросс-валидация модели"""
    print(f"\n🔄 Кросс-валидация (k={cv} фолдов)...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Разные метрики
    scoring_metrics = ['roc_auc', 'precision', 'recall', 'f1']
    results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
        results[metric] = scores
        print(f"   {metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return results

def check_data_leakage(df, feature_cols, target_col='is_fraud'):
    """Проверить на утечку данных"""
    print("\n🔍 Проверка на утечку данных...")
    
    suspicious_features = []
    
    for col in feature_cols:
        # Корреляция с целевой переменной
        if df[col].dtype in ['int64', 'float64']:
            corr = abs(df[col].corr(df[target_col].astype(int)))
            
            if corr > 0.95:  # Подозрительно высокая корреляция
                suspicious_features.append((col, corr))
                print(f"   ⚠️ {col}: корреляция = {corr:.4f}")
    
    if suspicious_features:
        print(f"\n   ⚠️ Найдено {len(suspicious_features)} подозрительных признаков!")
    else:
        print("   ✅ Утечки данных не обнаружено")
    
    return suspicious_features

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main():
    # Загрузка данных
    df = load_data_from_db()
    
    # Feature engineering
    df = engineer_features_advanced(df)
    
    # Определение признаков (ТЕ ЖЕ, ЧТО В train_improved_model.py)
    feature_cols = [
        # Основные признаки суммы
        'amount', 'amount_log', 'amount_sqrt', 'amount_cbrt', 'amount_category',
        'is_round_100', 'is_round_1000', 'is_round_10000',
        
        # Временные признаки
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
        'is_night', 'is_early_morning', 'is_morning', 'is_afternoon', 'is_evening', 
        'is_weekend', 'is_peak_fraud_hour',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        
        # Поведенческие паттерны
        'avg_logins_per_day_30d', 'avg_logins_per_day_7d',
        'avg_session_interval_sec', 'burstiness_score',
        'exp_weighted_avg_interval', 'fano_factor', 'interval_zscore',
        'login_freq_change_ratio', 'login_ratio_7d_30d',
        'logins_last_30_days', 'logins_last_7_days',
        'session_interval_std', 'session_interval_variance',
        'unique_os_versions_30d', 'unique_phone_models_30d',
        'has_os_data', 'has_phone_data', 'device_diversity', 'is_multi_device',
        'has_behavior_data', 'login_activity_score', 'is_zero_activity',
        'is_high_burstiness', 'is_unusual_interval',
        
        # Комбинированные признаки
        'night_large_amount', 'weekend_night', 'large_amount_no_activity',
        'round_amount_night'
    ]
    
    X = df[feature_cols]
    y = df['is_fraud'].astype(int)
    
    print(f"\n📊 Датасет:")
    print(f"   Размер: {len(df)} записей")
    print(f"   Признаков: {len(feature_cols)}")
    print(f"   Fraud: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Clean: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Проверка на утечку данных
    suspicious = check_data_leakage(df, feature_cols)
    
    # Загрузка обученной модели
    print("\n📦 Загрузка обученной модели...")
    
    # Автоопределение типа модели по существующим файлам
    model_files = {
        'LightGBM': Path('models/fraud_model_improved.txt'),
        'CatBoost': Path('models/fraud_model_improved.cbm'),
        'XGBoost': Path('models/fraud_model_improved.json')
    }
    
    model_type = None
    for mtype, mpath in model_files.items():
        if mpath.exists():
            model_type = mtype
            print(f"   📋 Найдена модель: {model_type}")
            break
    
    if not model_type:
        print("   ⚠️ Модель не найдена, создаём новую CatBoost...")
        model_type = 'CatBoost'
    
    try:
        if model_type == 'LightGBM':
            # Используем нашу обёртку
            model = LGBMWrapper('models/fraud_model_improved.txt')
            model.fit(X, y)  # Загрузит pre-trained модель
            print("   ✅ LightGBM модель загружена")
            
        elif model_type == 'CatBoost':
            model = cb.CatBoostClassifier()
            model.load_model('models/fraud_model_improved.cbm')
            print("   ✅ CatBoost модель загружена")
            
        elif model_type == 'XGBoost':
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model('models/fraud_model_improved.json')
            print("   ✅ XGBoost модель загружена")
            
    except Exception as e:
        print(f"   ⚠️ Не удалось загрузить модель: {e}")
        print("   Создаем новую CatBoost модель...")
        
        # Вес для балансировки классов
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        
        model = cb.CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            depth=8,
            learning_rate=0.05,
            iterations=300,
            l2_leaf_reg=3,
            subsample=0.8,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=False
        )
        model.fit(X, y)
    
    # Кросс-валидация
    cv_results = cross_validate_model(model, X, y, cv=5)
    
    # Обучение для визуализации (только для CatBoost)
    if model_type == 'CatBoost':
        print("\n🎨 Создание визуализаций...")
        model.fit(X, y)
        y_pred = model.predict(X).flatten()
        y_proba = model.predict_proba(X)[:, 1]
        
        # Создание папки для графиков
        import os
        os.makedirs('validation_plots', exist_ok=True)
        
        # 1. Confusion Matrix
        fig1 = plot_confusion_matrix(y, y_pred, "Confusion Matrix (Full Dataset)")
        fig1.savefig('validation_plots/01_confusion_matrix.png', dpi=150)
        plt.close()
        print("   ✅ Confusion Matrix сохранена")
        
        # 2. ROC Curve
        fig2 = plot_roc_curve(y, y_proba)
        fig2.savefig('validation_plots/02_roc_curve.png', dpi=150)
        plt.close()
        print("   ✅ ROC Curve сохранена")
        
        # 3. Precision-Recall Curve
        fig3 = plot_precision_recall_curve(y, y_proba)
        fig3.savefig('validation_plots/03_precision_recall.png', dpi=150)
        plt.close()
        print("   ✅ Precision-Recall Curve сохранена")
        
        # 4. Learning Curves
        fig4 = plot_learning_curves(model, X, y, cv=5)
        fig4.savefig('validation_plots/04_learning_curves.png', dpi=150)
        plt.close()
        print("   ✅ Learning Curves сохранены")
    else:
        print("\n⚠️ Визуализации доступны только для CatBoost модели")
    
    # Итоговый отчет
    print("\n" + "=" * 70)
    print("📋 ИТОГОВЫЙ ОТЧЕТ ВАЛИДАЦИИ")
    print("=" * 70)
    
    print("\n1️⃣ КРОСС-ВАЛИДАЦИЯ (5 фолдов):")
    for metric, scores in cv_results.items():
        print(f"   {metric.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    print("\n2️⃣ АНАЛИЗ РЕЗУЛЬТАТОВ:")
    recall_mean = cv_results['recall'].mean()
    precision_mean = cv_results['precision'].mean()
    f1_mean = cv_results['f1'].mean()
    
    if recall_mean < 0.5:
        print(f"   ⚠️ RECALL НИЗКИЙ ({recall_mean:.1%})")
        print(f"      Модель пропускает {(1-recall_mean)*100:.0f}% мошенников!")
        print(f"      Рекомендации:")
        print(f"      - Увеличить sampling_strategy до 0.8")
        print(f"      - Снизить порог классификации до 0.3")
        print(f"      - Использовать CatBoost вместо LightGBM")
    elif recall_mean < 0.7:
        print(f"   ⚡ RECALL СРЕДНИЙ ({recall_mean:.1%})")
        print(f"      Можно улучшить снижением порога или увеличением SMOTE")
    else:
        print(f"   ✅ RECALL ХОРОШИЙ ({recall_mean:.1%})")
    
    if precision_mean < 0.5:
        print(f"   ⚠️ PRECISION НИЗКИЙ ({precision_mean:.1%})")
        print(f"      Много ложных срабатываний")
    elif precision_mean < 0.7:
        print(f"   ⚡ PRECISION СРЕДНИЙ ({precision_mean:.1%})")
    else:
        print(f"   ✅ PRECISION ХОРОШИЙ ({precision_mean:.1%})")
    
    print(f"\n   F1-SCORE: {f1_mean:.4f}")
    
    print("\n3️⃣ ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ:")
    train_score = cv_results['roc_auc'].mean()
    if train_score > 0.95:
        print("   ⚡ Умеренный риск переобучения")
        print(f"   - AUC = {train_score:.4f}")
    else:
        print("   ✅ Переобучение не обнаружено")
    
    print("\n4️⃣ РЕКОМЕНДАЦИИ:")
    if len(suspicious) > 0:
        print("   ⚠️ Обнаружены подозрительные признаки - проверьте утечку данных")
    if y.sum() < 100:
        print("   ⚠️ Мало примеров fraud - соберите больше данных")
    if recall_mean < 0.5:
        print("   🔧 СРОЧНО: Нужно улучшить Recall!")
        print("      1. Запусти переобучение с sampling_strategy=0.8")
        print("      2. Используй CatBoost (он показал лучший Recall)")
        print("      3. Снизь порог классификации до 0.3")
    
    if model_type == 'CatBoost':
        print("\n📁 Все графики сохранены в папке: validation_plots/")
    
    print("\n" + "=" * 70)
    print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)

if __name__ == "__main__":
    main()