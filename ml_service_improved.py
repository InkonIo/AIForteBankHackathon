"""
🚀 ML API СЕРВИС ДЛЯ ФРОНТЕНДА
Принимает транзакции и возвращает решение модели
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Разрешаем запросы с фронтенда

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_db',
    'user': 'postgres',
    'password': 'Alikhancool20!'
}

# Загрузка модели
print("🔄 Загрузка ML модели...")
try:
    model = xgb.Booster()
    model.load_model('models/fraud_model.json')
    print("✅ Модель загружена успешно!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

# Пороги решений
THRESHOLDS = {
    'approve_max': 0.3,
    'verify_max': 0.5,
    'review_max': 0.7,
    'block_min': 0.85
}

# Список признаков (должен совпадать с обучением)
FEATURE_COLUMNS = [
    'amount', 'amount_log', 'amount_sqrt',
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
    'is_night', 'is_morning', 'is_evening', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'avg_logins_per_day_30d', 'avg_logins_per_day_7d',
    'avg_session_interval_sec', 'burstiness_score',
    'exp_weighted_avg_interval', 'fano_factor', 'interval_zscore',
    'login_freq_change_ratio', 'login_ratio_7d_30d',
    'logins_last_30_days', 'logins_last_7_days',
    'session_interval_std', 'session_interval_variance',
    'unique_os_versions_30d', 'unique_phone_models_30d',
    'has_os_data', 'has_phone_data'
]


def get_customer_behavior(customer_id, trans_date):
    """Получить поведенческие паттерны клиента"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                avg_logins_per_day_30d,
                avg_logins_per_day_7d,
                avg_session_interval_sec,
                burstiness_score,
                exp_weighted_avg_interval,
                fano_factor,
                interval_zscore,
                latest_os_version,
                latest_phone_model,
                login_freq_change_ratio,
                login_ratio_7d_30d,
                logins_last_30_days,
                logins_last_7_days,
                session_interval_std,
                session_interval_variance,
                unique_os_versions_30d,
                unique_phone_models_30d
            FROM customer_behavior_patterns
            WHERE customer_id = %s
            AND trans_date = %s
            ORDER BY trans_date DESC
            LIMIT 1
        """
        
        cursor.execute(query, (customer_id, trans_date))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'avg_logins_per_day_30d': float(row[0]) if row[0] is not None else 0.0,
                'avg_logins_per_day_7d': float(row[1]) if row[1] is not None else 0.0,
                'avg_session_interval_sec': float(row[2]) if row[2] is not None else 0.0,
                'burstiness_score': float(row[3]) if row[3] is not None else 0.0,
                'exp_weighted_avg_interval': float(row[4]) if row[4] is not None else 0.0,
                'fano_factor': float(row[5]) if row[5] is not None else 0.0,
                'interval_zscore': float(row[6]) if row[6] is not None else 0.0,
                'latest_os_version': row[7],
                'latest_phone_model': row[8],
                'login_freq_change_ratio': float(row[9]) if row[9] is not None else 0.0,
                'login_ratio_7d_30d': float(row[10]) if row[10] is not None else 0.0,
                'logins_last_30_days': int(row[11]) if row[11] is not None else 0,
                'logins_last_7_days': int(row[12]) if row[12] is not None else 0,
                'session_interval_std': float(row[13]) if row[13] is not None else 0.0,
                'session_interval_variance': float(row[14]) if row[14] is not None else 0.0,
                'unique_os_versions_30d': int(row[15]) if row[15] is not None else 0,
                'unique_phone_models_30d': int(row[16]) if row[16] is not None else 0
            }
        else:
            # Если данных нет - вернуть нули
            return {
                'avg_logins_per_day_30d': 0.0,
                'avg_logins_per_day_7d': 0.0,
                'avg_session_interval_sec': 0.0,
                'burstiness_score': 0.0,
                'exp_weighted_avg_interval': 0.0,
                'fano_factor': 0.0,
                'interval_zscore': 0.0,
                'latest_os_version': None,
                'latest_phone_model': None,
                'login_freq_change_ratio': 0.0,
                'login_ratio_7d_30d': 0.0,
                'logins_last_30_days': 0,
                'logins_last_7_days': 0,
                'session_interval_std': 0.0,
                'session_interval_variance': 0.0,
                'unique_os_versions_30d': 0,
                'unique_phone_models_30d': 0
            }
    except Exception as e:
        print(f"Ошибка получения поведения: {e}")
        return None


def prepare_features(transaction_data):
    """Подготовить признаки для модели"""
    
    # Основные данные транзакции
    amount = float(transaction_data['amount'])
    trans_datetime = datetime.fromisoformat(transaction_data['datetime'].replace('Z', '+00:00'))
    customer_id = transaction_data['customer_id']
    trans_date = trans_datetime.date()
    
    # Временные признаки
    hour = trans_datetime.hour
    minute = trans_datetime.minute
    day_of_week = trans_datetime.weekday()
    day_of_month = trans_datetime.day
    month = trans_datetime.month
    
    # Получить поведенческие паттерны
    behavior = get_customer_behavior(customer_id, trans_date)
    if not behavior:
        behavior = {
            'avg_logins_per_day_30d': 0.0,
            'avg_logins_per_day_7d': 0.0,
            'avg_session_interval_sec': 0.0,
            'burstiness_score': 0.0,
            'exp_weighted_avg_interval': 0.0,
            'fano_factor': 0.0,
            'interval_zscore': 0.0,
            'login_freq_change_ratio': 0.0,
            'login_ratio_7d_30d': 0.0,
            'logins_last_30_days': 0,
            'logins_last_7_days': 0,
            'session_interval_std': 0.0,
            'session_interval_variance': 0.0,
            'unique_os_versions_30d': 0,
            'unique_phone_models_30d': 0,
            'latest_os_version': None,
            'latest_phone_model': None
        }
    
    # Создать словарь признаков
    features = {
        # Основные признаки суммы
        'amount': float(amount),
        'amount_log': float(np.log(amount + 1)),
        'amount_sqrt': float(np.sqrt(amount)),
        
        # Временные признаки
        'hour': int(hour),
        'minute': int(minute),
        'day_of_week': int(day_of_week),
        'day_of_month': int(day_of_month),
        'month': int(month),
        
        # Временные флаги
        'is_night': int(1 if (hour >= 23 or hour < 6) else 0),
        'is_morning': int(1 if (6 <= hour < 12) else 0),
        'is_evening': int(1 if (18 <= hour < 23) else 0),
        'is_weekend': int(1 if day_of_week in [5, 6] else 0),
        
        # Циклические признаки
        'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
        'hour_cos': float(np.cos(2 * np.pi * hour / 24)),
        'day_sin': float(np.sin(2 * np.pi * day_of_week / 7)),
        'day_cos': float(np.cos(2 * np.pi * day_of_week / 7)),
        
        # Поведенческие паттерны
        'avg_logins_per_day_30d': float(behavior['avg_logins_per_day_30d']),
        'avg_logins_per_day_7d': float(behavior['avg_logins_per_day_7d']),
        'avg_session_interval_sec': float(behavior['avg_session_interval_sec']),
        'burstiness_score': float(behavior['burstiness_score']),
        'exp_weighted_avg_interval': float(behavior['exp_weighted_avg_interval']),
        'fano_factor': float(behavior['fano_factor']),
        'interval_zscore': float(behavior['interval_zscore']),
        'login_freq_change_ratio': float(behavior['login_freq_change_ratio']),
        'login_ratio_7d_30d': float(behavior['login_ratio_7d_30d']),
        'logins_last_30_days': int(behavior['logins_last_30_days']),
        'logins_last_7_days': int(behavior['logins_last_7_days']),
        'session_interval_std': float(behavior['session_interval_std']),
        'session_interval_variance': float(behavior['session_interval_variance']),
        'unique_os_versions_30d': int(behavior['unique_os_versions_30d']),
        'unique_phone_models_30d': int(behavior['unique_phone_models_30d']),
        
        # Категориальные признаки
        'has_os_data': int(1 if behavior['latest_os_version'] else 0),
        'has_phone_data': int(1 if behavior['latest_phone_model'] else 0)
    }
    
    return features


def get_decision(probability):
    """Определить решение на основе вероятности"""
    if probability < THRESHOLDS['approve_max']:
        return 'APPROVE', '✅ Одобрена', 'Низкий риск мошенничества'
    elif probability < THRESHOLDS['verify_max']:
        return 'VERIFY', '🔍 Требуется проверка', 'Средний риск - нужна доп. верификация'
    elif probability < THRESHOLDS['review_max']:
        return 'REVIEW', '⚠️ Ручная проверка', 'Высокий риск - требуется анализ'
    else:
        return 'BLOCK', '🚫 Заблокирована', 'Критический риск мошенничества'


def get_feature_importance():
    """Получить важность признаков"""
    try:
        importance = model.get_score(importance_type='weight')
        # Сортируем по важности
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance[:10]  # Топ-10
    except:
        return []


@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание для транзакции"""
    
    if not model:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    try:
        # Получить данные транзакции
        data = request.json
        
        # Подготовить признаки
        features = prepare_features(data)
        
        # Создать DataFrame с признаками в правильном порядке
        X = pd.DataFrame([features])[FEATURE_COLUMNS]
        
        # КРИТИЧНО: Конвертировать все столбцы в float
        # БД возвращает Decimal, который XGBoost не понимает
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
        
        # Создать DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
        
        # Предсказание
        fraud_probability = float(model.predict(dmatrix)[0])
        
        # Определить решение
        decision, decision_label, decision_reason = get_decision(fraud_probability)
        
        # Получить важность признаков для объяснения
        feature_importance = get_feature_importance()
        
        # Топ-5 факторов риска
        risk_factors = []
        for feature_name in ['is_night', 'amount_log', 'burstiness_score', 'interval_zscore', 'logins_last_7_days']:
            if feature_name in features:
                risk_factors.append({
                    'name': feature_name,
                    'value': features[feature_name],
                    'impact': 'increase' if features[feature_name] > 0 else 'decrease'
                })
        
        # Формируем ответ
        response = {
            'fraud_probability': round(fraud_probability * 100, 2),
            'decision': decision,
            'decision_label': decision_label,
            'decision_reason': decision_reason,
            'risk_factors': risk_factors[:5],
            'feature_importance': [
                {'feature': f[0], 'importance': int(f[1])} 
                for f in feature_importance[:8]
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Статистика модели"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Общее количество транзакций
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0]
        
        # Мошеннические транзакции
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")
        fraud_transactions = cursor.fetchone()[0]
        
        # Заблокированные (если есть поле status)
        blocked = 0
        
        conn.close()
        
        return jsonify({
            'total_transactions': total_transactions,
            'fraud_detected': fraud_transactions,
            'blocked': blocked,
            'accuracy': 0.86,  # Из метрик обучения
            'precision': 0.96,
            'recall': 0.88,
            'f1_score': 0.92
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_sample_transaction', methods=['GET'])
def get_sample_transaction():
    """Получить случайную транзакцию из БД"""
    try:
        # Параметр: fraud или clean
        trans_type = request.args.get('type', 'random')
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        if trans_type == 'fraud':
            # Мошенническая транзакция
            query = """
                SELECT 
                    transaction_id,
                    customer_id,
                    recipient_id,
                    amount,
                    transaction_datetime,
                    is_fraud
                FROM transactions
                WHERE is_fraud = true
                ORDER BY RANDOM()
                LIMIT 1
            """
        elif trans_type == 'clean':
            # Чистая транзакция
            query = """
                SELECT 
                    transaction_id,
                    customer_id,
                    recipient_id,
                    amount,
                    transaction_datetime,
                    is_fraud
                FROM transactions
                WHERE is_fraud = false
                ORDER BY RANDOM()
                LIMIT 1
            """
        else:
            # Любая случайная
            query = """
                SELECT 
                    transaction_id,
                    customer_id,
                    recipient_id,
                    amount,
                    transaction_datetime,
                    is_fraud
                FROM transactions
                ORDER BY RANDOM()
                LIMIT 1
            """
        
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                'transaction_id': row[0],
                'customer_id': row[1],
                'recipient_id': row[2],
                'amount': float(row[3]),
                'datetime': row[4].isoformat(),
                'is_fraud': row[5],
                'actual_label': 'Мошенничество' if row[5] else 'Чистая'
            })
        else:
            return jsonify({'error': 'Нет транзакций'}), 404
            
    except Exception as e:
        print(f"Ошибка получения транзакции: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ML API СЕРВИС ЗАПУЩЕН")
    print("="*70)
    print(f"📡 API доступен на: http://localhost:5001")
    print(f"🔗 Endpoints:")
    print(f"   GET  /health   - Проверка работоспособности")
    print(f"   POST /predict  - Предсказание для транзакции")
    print(f"   GET  /stats    - Статистика модели")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)