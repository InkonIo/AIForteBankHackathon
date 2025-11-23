"""
🎯 ОБУЧЕНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ ИЗ БД
Использует transactions + customer_behavior_patterns
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from pathlib import Path
import json

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_db',
    'user': 'postgres',
    'password': 'Alikhancool20!'
}

print("="*70)
print("🎯 ОБУЧЕНИЕ ML-МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
print("="*70)

# ==================== ЗАГРУЗКА ДАННЫХ ИЗ БД ====================

def load_training_data():
    """Загрузить данные для обучения из БД"""
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    print("\n📥 Загрузка данных из БД...")
    
    # Запрос который соединяет transactions с customer_behavior_patterns
    # Используем РЕАЛЬНЫЕ столбцы из таблицы customer_behavior_patterns
    query = """
    SELECT 
        t.id,
        t.transaction_id,
        t.customer_id,
        t.recipient_id,
        t.amount,
        t.transaction_datetime,
        t.is_fraud,
        
        -- Извлекаем временные признаки
        EXTRACT(HOUR FROM t.transaction_datetime) as hour,
        EXTRACT(MINUTE FROM t.transaction_datetime) as minute,
        EXTRACT(DOW FROM t.transaction_datetime) as day_of_week,
        EXTRACT(DAY FROM t.transaction_datetime) as day_of_month,
        EXTRACT(MONTH FROM t.transaction_datetime) as month,
        
        -- Поведенческие паттерны из customer_behavior_patterns (РЕАЛЬНЫЕ столбцы)
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
    WHERE t.transaction_datetime >= NOW() - INTERVAL '90 days'
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Загружено {len(df)} транзакций")
    print(f"   Мошеннических: {df['is_fraud'].sum()}")
    print(f"   Чистых: {(~df['is_fraud']).sum()}")
    
    return df

# ==================== FEATURE ENGINEERING ====================

def engineer_features(df):
    """Создать дополнительные признаки"""
    
    print("\n🔧 Feature Engineering...")
    
    # Копируем для безопасности
    df = df.copy()
    
    # 1. Логарифмические признаки суммы
    df['amount_log'] = np.log(df['amount'] + 1)
    df['amount_sqrt'] = np.sqrt(df['amount'])
    
    # 2. Временные признаки
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    
    # 3. Циклические признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 4. Кодируем категориальные признаки (OS и Phone)
    # Просто проверяем наличие данных
    df['has_os_data'] = (~df['latest_os_version'].isna()).astype(int)
    df['has_phone_data'] = (~df['latest_phone_model'].isna()).astype(int)
    
    # Заполнить пропуски нулями
    df = df.fillna(0)
    
    # Убрать inf значения
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Создано {df.shape[1]} признаков")
    
    return df

# ==================== ОБУЧЕНИЕ МОДЕЛИ ====================

# Все признаки для модели (ТОЛЬКО существующие в БД + созданные)
FEATURE_COLUMNS = [
    # Основные признаки транзакции
    'amount', 'amount_log', 'amount_sqrt',
    
    # Временные признаки
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
    'is_night', 'is_morning', 'is_evening', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    
    # Поведенческие паттерны (из customer_behavior_patterns)
    'avg_logins_per_day_30d',
    'avg_logins_per_day_7d',
    'avg_session_interval_sec',
    'burstiness_score',
    'exp_weighted_avg_interval',
    'fano_factor',
    'interval_zscore',
    'login_freq_change_ratio',
    'login_ratio_7d_30d',
    'logins_last_30_days',
    'logins_last_7_days',
    'session_interval_std',
    'session_interval_variance',
    'unique_os_versions_30d',
    'unique_phone_models_30d',
    'has_os_data',
    'has_phone_data'
]

def train_model(df):
    """Обучить XGBoost модель"""
    
    print("\n🧠 Обучение XGBoost модели...")
    
    # Подготовить данные
    X = df[FEATURE_COLUMNS]
    y = df['is_fraud'].astype(int)
    
    # Разделить на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"   Test:  {len(X_test)} ({y_test.sum()} fraud)")
    
    # Создать DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)
    
    # Параметры модели (оптимизированы для fraud detection)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'eta': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),  # Баланс классов
        'seed': 42
    }
    
    print(f"   Scale pos weight: {params['scale_pos_weight']:.2f}")
    
    # Обучение с early stopping
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Предсказания
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Метрики
    print("\n" + "="*70)
    print("📊 МЕТРИКИ МОДЕЛИ")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Clean', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc:.4f}")
    
    # Оптимальные пороги
    print("\n" + "="*70)
    print("🎯 ОПТИМАЛЬНЫЕ ПОРОГИ")
    print("="*70)
    
    thresholds = find_optimal_thresholds(y_test, y_pred_proba)
    
    for name, value in thresholds.items():
        print(f"   {name:20s}: {value:.4f}")
    
    return model, thresholds

def find_optimal_thresholds(y_test, y_pred_proba):
    """Найти оптимальные пороги для разных уровней риска"""
    
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    
    # F1 score для каждого порога
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Оптимальный порог по F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    
    print(f"\nОптимальный порог (по F1): {optimal_threshold:.4f}")
    print(f"   Precision: {precision[optimal_idx]:.4f}")
    print(f"   Recall: {recall[optimal_idx]:.4f}")
    print(f"   F1: {f1_scores[optimal_idx]:.4f}")
    
    # Градация порогов
    return {
        'approve_max': 0.3,  # Ниже - автоматическое одобрение
        'verify_max': 0.5,   # Требуется доп. верификация
        'review_max': 0.7,   # Требуется ручная проверка
        'block_min': 0.85    # Автоматическая блокировка
    }

# ==================== СОХРАНЕНИЕ ====================

def save_model(model, thresholds):
    """Сохранить модель и пороги"""
    
    print("\n💾 Сохранение модели...")
    
    # Создать директорию
    Path('models').mkdir(exist_ok=True)
    
    # Сохранить модель
    model.save_model('models/fraud_model.json')
    print("   ✅ Модель сохранена: models/fraud_model.json")
    
    # Сохранить пороги
    with open('models/thresholds.txt', 'w') as f:
        f.write(f"APPROVE (max): {thresholds['approve_max']:.4f}\n")
        f.write(f"VERIFY (max): {thresholds['verify_max']:.4f}\n")
        f.write(f"REVIEW (max): {thresholds['review_max']:.4f}\n")
        f.write(f"BLOCK (min): {thresholds['block_min']:.4f}\n")
    
    print("   ✅ Пороги сохранены: models/thresholds.txt")
    
    # Сохранить список признаков
    with open('models/feature_columns.txt', 'w', encoding='utf-8') as f:
        f.write("FEATURE_COLUMNS = [\n")
        for col in FEATURE_COLUMNS:
            f.write(f"    '{col}',\n")
        f.write("]\n")
    
    print("   ✅ Признаки сохранены: models/feature_columns.txt")
    
    # Сохранить метрики
    with open('models/metrics.txt', 'w', encoding='utf-8') as f:
        f.write("Модель обучена на реальных данных из БД\n")
        f.write(f"Дата обучения: {pd.Timestamp.now()}\n")
        f.write(f"Количество признаков: {len(FEATURE_COLUMNS)}\n")
    
    print("   ✅ Метрики сохранены: models/metrics.txt")

# ==================== MAIN ====================

def main():
    try:
        # 1. Загрузить данные
        df = load_training_data()
        
        if len(df) < 100:
            print("\n⚠️  ВНИМАНИЕ: Мало данных для обучения!")
            print(f"   Загружено {len(df)} транзакций (рекомендуется >100)")
            print("   Попробуем обучить модель на имеющихся данных...")
        
        if df['is_fraud'].sum() < 10:
            print("\n⚠️  ВНИМАНИЕ: Мало примеров мошенничества!")
            print(f"   Найдено {df['is_fraud'].sum()} мошеннических транзакций (рекомендуется >10)")
            print("   Попробуем обучить модель на имеющихся данных...")
        
        # 2. Feature Engineering
        df = engineer_features(df)
        
        # 3. Обучить модель
        model, thresholds = train_model(df)
        
        # 4. Сохранить
        save_model(model, thresholds)
        
        print("\n" + "="*70)
        print("✅ ГОТОВО!")
        print("="*70)
        print("\n🚀 Теперь перезапустите ML API сервис:")
        print("   python ml_service_improved.py")
        print("\n📊 Модель готова к использованию!")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()