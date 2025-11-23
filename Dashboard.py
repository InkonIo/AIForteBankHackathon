"""
🛡️ ML Fraud Detection Dashboard
Интерактивная система детекции мошенничества
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from datetime import datetime, timedelta
import json
import openai
from typing import Dict, List

import time
import random

# ==================== КОНФИГУРАЦИЯ ====================

st.set_page_config(
    page_title="🛡️ Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .fraud-card {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .clean-card {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .review-card {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_db',
    'user': 'postgres',
    'password': 'Alikhancool20!'
}

ML_API_URL = "http://localhost:5000"

# Замените функции подключения к БД на эти:

# ==================== ФУНКЦИИ БД (ИСПРАВЛЕНО) ====================

def get_db_connection():
    """Подключение к PostgreSQL - НЕ кешируем!"""
    return psycopg2.connect(**DB_CONFIG)

@st.cache_data(ttl=60)
def load_transactions(limit=100):
    """Загрузить транзакции"""
    conn = None
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            id,
            transaction_id,
            customer_id,
            recipient_id,
            amount,
            transaction_datetime,
            is_fraud,
            fraud_probability,
            status,
            created_at
        FROM transactions
        ORDER BY transaction_datetime DESC
        LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=60)
def load_statistics():
    """Статистика по транзакциям"""
    conn = None
    try:
        conn = get_db_connection()
        
        query = """
        SELECT 
            COUNT(*) as total_transactions,
            COUNT(*) FILTER (WHERE is_fraud = true) as fraud_count,
            COUNT(*) FILTER (WHERE is_fraud = false) as clean_count,
            AVG(amount) as avg_amount,
            SUM(amount) as total_amount,
            SUM(amount) FILTER (WHERE is_fraud = true) as fraud_amount_saved,
            COUNT(DISTINCT customer_id) as unique_customers
        FROM transactions
        """
        
        df = pd.read_sql(query, conn)
        return df.iloc[0].to_dict()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=60)
def load_hourly_stats():
    """Статистика по часам"""
    conn = None
    try:
        conn = get_db_connection()
        
        query = """
        SELECT 
            EXTRACT(HOUR FROM transaction_datetime) as hour,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE is_fraud = true) as fraud
        FROM transactions
        GROUP BY hour
        ORDER BY hour
        """
        
        df = pd.read_sql(query, conn)
        return df
    finally:
        if conn:
            conn.close()

def get_transaction_by_id(transaction_id):
    """Получить транзакцию по ID"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT * FROM transactions WHERE id = %s
        """
        
        cursor.execute(query, (transaction_id,))
        result = cursor.fetchone()
        
        return dict(result) if result else None
    finally:
        if conn:
            conn.close()

def get_customer_behavior(customer_id, trans_date):
    """Получить поведенческие паттерны клиента"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT * FROM customer_behavior_patterns 
        WHERE customer_id = %s AND trans_date = %s
        """
        
        cursor.execute(query, (customer_id, trans_date))
        result = cursor.fetchone()
        
        return dict(result) if result else None
    finally:
        if conn:
            conn.close()

def update_transaction_status(transaction_id, new_status, decision_maker="analyst"):
    """Обновить статус транзакции"""
    conn = None
    try:
        # Нормализовать статус под БД constraint
        status_map = {
            'block': 'blocked',
            'blocked': 'blocked',
            'approve': 'approved',
            'approved': 'approved',
            'review': 'review',
            'pending': 'pending'
        }
        
        # Преобразовать статус
        normalized_status = status_map.get(new_status.lower(), 'pending')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        UPDATE transactions 
        SET status = %s, updated_at = NOW()
        WHERE id = %s
        """
        
        cursor.execute(query, (normalized_status, transaction_id))
        conn.commit()
        return True  # ✅ Успех
    except Exception as e:
        st.error(f"❌ Ошибка обновления статуса: {e}")
        return False  # ❌ Провал
    finally:
        if conn:
            conn.close()

# ==================== ML API ====================

def call_ml_api(features):
    """Вызвать ML API для предсказания"""
    try:
        response = requests.post(
            f"{ML_API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ ML API вернул ошибку: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ ML сервис недоступен! Запустите: python ml_service.py")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None

def extract_features_from_transaction(trans, behavior=None):
    """Извлечь признаки из транзакции"""
    amount = float(trans['amount'])
    dt = pd.to_datetime(trans['transaction_datetime'])
    
    features = {
        # Базовые признаки
        'amount': amount,
        'amount_log': np.log1p(amount),
        'amount_sqrt': np.sqrt(amount),
        
        # Временные признаки
        'hour': dt.hour,
        'minute': dt.minute,
        'day_of_week': dt.dayofweek + 1,
        'day_of_month': dt.day,
        'month': dt.month,
        'is_night': 1.0 if 0 <= dt.hour < 6 else 0.0,
        'is_morning': 1.0 if 6 <= dt.hour < 12 else 0.0,
        'is_evening': 1.0 if 18 <= dt.hour < 24 else 0.0,
        'is_weekend': 1.0 if dt.dayofweek >= 5 else 0.0,
        
        # Циклические признаки
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        'day_sin': np.sin(2 * np.pi * dt.dayofweek / 7),
        'day_cos': np.cos(2 * np.pi * dt.dayofweek / 7),
    }
    
    # Добавить поведенческие признаки если есть
    if behavior:
        features.update({
            'unique_os_30d': float(behavior.get('unique_os_versions_30d', 0) or 0),
            'unique_phones_30d': float(behavior.get('unique_phone_models_30d', 0) or 0),
            'logins_7d': float(behavior.get('logins_last_7_days', 0) or 0),
            'logins_30d': float(behavior.get('logins_last_30_days', 0) or 0),
            'avg_logins_per_day_7d': float(behavior.get('avg_logins_per_day_7d', 0) or 0),
            'avg_logins_per_day_30d': float(behavior.get('avg_logins_per_day_30d', 0) or 0),
            'login_freq_change_ratio': float(behavior.get('login_freq_change_ratio', 0) or 0),
            'login_ratio_7d_30d': float(behavior.get('login_ratio_7d_30d', 0) or 0),
            'avg_session_interval_sec': float(behavior.get('avg_session_interval_sec', 0) or 0),
            'session_interval_std': float(behavior.get('session_interval_std', 0) or 0),
            'session_interval_variance': float(behavior.get('session_interval_variance', 0) or 0),
            'exp_weighted_avg_interval': float(behavior.get('exp_weighted_avg_interval', 0) or 0),
            'burstiness_score': float(behavior.get('burstiness_score', 0) or 0),
            'fano_factor': float(behavior.get('fano_factor', 0) or 0),
            'interval_zscore': float(behavior.get('interval_zscore', 0) or 0),
        })
    else:
        # Заполнить нулями если нет данных
        for key in ['unique_os_30d', 'unique_phones_30d', 'logins_7d', 'logins_30d',
                    'avg_logins_per_day_7d', 'avg_logins_per_day_30d', 
                    'login_freq_change_ratio', 'login_ratio_7d_30d',
                    'avg_session_interval_sec', 'session_interval_std',
                    'session_interval_variance', 'exp_weighted_avg_interval',
                    'burstiness_score', 'fano_factor', 'interval_zscore']:
            features[key] = 0.0
    
    # Упрощенные признаки (без истории)
    for key in ['trans_count_1h', 'trans_count_24h', 'trans_count_7d', 'trans_count_30d',
                'avg_amount_30d', 'amount_ratio_to_avg', 'amount_std_30d', 'max_amount_30d',
                'hours_since_last_trans', 'is_new_customer', 'is_new_recipient',
                'trans_to_recipient_count', 'unique_recipients_7d', 'unique_recipients_30d']:
        features[key] = 0.0
    
    return features

# ==================== ГЛАВНОЕ ПРИЛОЖЕНИЕ ====================


# ==================== AI SERVICE ====================

class AIRecommendationService:
    def __init__(self, api_key: str = None):
        """Инициализация с API ключом"""
        self.api_key = api_key
        self.enabled = api_key is not None and api_key.startswith('sk-')
        if self.enabled:
            openai.api_key = api_key
    
    def generate_recommendation(
        self, 
        transaction: Dict,
        prediction: Dict,
        top_factors: List[Dict]
    ) -> str:
        """Генерирует AI рекомендацию"""
        if not self.enabled:
            return self._generate_fallback_recommendation(
                prediction.get('action', 'REVIEW'),
                prediction.get('probability', 0),
                top_factors
            )
        
        try:
            prompt = self._build_prompt(transaction, prediction, top_factors)
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Ты - эксперт по анти-фроду в банке. Объясняй решения простым языком."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            recommendation = response.choices[0].message.content.strip()
            return "🤖 **AI Рекомендация:**\n\n" + recommendation
            
        except Exception as e:
            return self._generate_fallback_recommendation(
                prediction.get('action', 'REVIEW'),
                prediction.get('probability', 0),
                top_factors
            )
    
    def _build_prompt(self, transaction: Dict, prediction: Dict, top_factors: List[Dict]) -> str:
        """Построить промпт"""
        amount = transaction.get('amount', 0)
        probability = prediction.get('probability', 0) * 100
        decision = prediction.get('action', 'REVIEW')
        risk_level = prediction.get('riskLevel', 'UNKNOWN')
        
        factors_text = "\n".join([
            f"- {f['featureName']}: {f['impact']:.3f} ({'увеличивает' if f['impact'] > 0 else 'снижает'} риск)"
            for f in top_factors[:3]
        ])
        
        return f"""
Проанализируй транзакцию и объясни решение ML модели простым языком.

ТРАНЗАКЦИЯ:
- Сумма: {amount:,.2f} ₸

ML АНАЛИЗ:
- Вероятность мошенничества: {probability:.1f}%
- Уровень риска: {risk_level}
- Рекомендуемое действие: {decision}

КЛЮЧЕВЫЕ ФАКТОРЫ:
{factors_text}

Напиши короткую рекомендацию (2-3 предложения):
1. Почему модель приняла это решение
2. На что обратить внимание
3. Какие действия предпринять
"""
    
    def _generate_fallback_recommendation(self, decision: str, probability: float, top_factors: List[Dict]) -> str:
        """Простая рекомендация без AI"""
        if decision == "BLOCK":
            rec = f"""
🚨 **ВЫСОКИЙ РИСК** ({probability*100:.1f}%)

**Рекомендации:**
1. Немедленно связаться с клиентом для подтверждения
2. Проверить историю транзакций за последние 24 часа
3. Убедиться в легитимности получателя
"""
        elif decision == "REVIEW":
            rec = f"""
⚠️ **СРЕДНИЙ РИСК** ({probability*100:.1f}%)

**Рекомендации:**
1. Проверить обычную активность клиента
2. Сравнить сумму со средними транзакциями
3. При сомнениях связаться с клиентом
"""
        else:
            rec = f"""
✅ **НИЗКИЙ РИСК** ({probability*100:.1f}%)

**Рекомендации:**
1. Стандартный мониторинг
2. Логирование для статистики
"""
        
        rec += "\n\n**Ключевые факторы:**\n"
        for factor in top_factors[:3]:
            direction = "↑" if factor['impact'] > 0 else "↓"
            rec += f"{direction} {factor['featureName']}\n"
        
        return rec
    
# ==================== НАСТРОЙКИ AI (в начале файла после DB_CONFIG) ====================

# AI Configuration
OPENAI_API_KEY = st.secrets.get("openai_api_key", None) if hasattr(st, 'secrets') else None

# Или загрузить из переменной окружения
if not OPENAI_API_KEY:
    import os
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)

# Инициализировать AI сервис
ai_service = AIRecommendationService(OPENAI_API_KEY)

"""
ПРАВИЛЬНЫЙ симулятор - берёт СЛУЧАЙНЫЕ транзакции из ВСЕЙ базы
Добавьте этот код в Dashboard вместо старого симулятора
"""

# ==================== ИСПРАВЛЕННЫЙ СИМУЛЯТОР ====================

def load_random_transactions_for_simulation(count=10):
    """
    Загрузить СЛУЧАЙНЫЕ транзакции из ВСЕЙ базы
    С реалистичным соотношением: ~99% чистых, ~1% фрод
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Получить общее количество транзакций
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total = cursor.fetchone()['count']
        
        if total == 0:
            return []
        
        # Получить количество фродов
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")
        total_fraud = cursor.fetchone()['count']
        
        # Рассчитать сколько фродов взять (реалистично: 0-1 из 10)
        fraud_rate = total_fraud / total if total > 0 else 0
        fraud_to_take = 1 if fraud_rate > 0.01 else 0  # Берём 1 фрод если есть
        clean_to_take = count - fraud_to_take
        
        transactions = []
        
        # 1. Взять случайные ЧИСТЫЕ транзакции
        query_clean = """
        SELECT * FROM transactions 
        WHERE is_fraud = false
        ORDER BY RANDOM()
        LIMIT %s
        """
        cursor.execute(query_clean, (clean_to_take,))
        transactions.extend(cursor.fetchall())
        
        # 2. Взять случайные ФРОД транзакции (если нужно)
        if fraud_to_take > 0:
            query_fraud = """
            SELECT * FROM transactions 
            WHERE is_fraud = true
            ORDER BY RANDOM()
            LIMIT %s
            """
            cursor.execute(query_fraud, (fraud_to_take,))
            transactions.extend(cursor.fetchall())
        
        # Перемешать
        import random
        random.shuffle(transactions)
        
        return [dict(t) for t in transactions]
        
    finally:
        if conn:
            conn.close()


def show_simulator():
    """ИСПРАВЛЕННАЯ версия симулятора входящих транзакций"""
    st.header("📨 Симулятор входящих транзакций")
    
    st.info("""
    🎯 **Симулятор:**
    - Берёт **случайные** транзакции из всей базы
    - Реалистичное соотношение: ~9 чистых, ~1 фрод
    - ML модель анализирует каждую
    - Автоматически применяет решение
    """)
    
    # Настройки
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "🎯 Порог блокировки (вероятность фрода)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Транзакции с вероятностью выше этого порога будут заблокированы"
        )
    
    with col2:
        delay = st.slider(
            "⏱️ Задержка между транзакциями (сек)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    auto_mode = st.checkbox("🤖 Автоматический режим", value=True, 
                           help="Автоматически применять решения ML модели")
    
    # Инициализация статистики
    if 'sim_stats' not in st.session_state:
        st.session_state.sim_stats = {
            'total': 0, 'blocked': 0, 'approved': 0, 
            'reviewed': 0, 'fraud_caught': 0, 'fraud_missed': 0
        }
    
    # Показать статистику
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Всего проверено", st.session_state.sim_stats['total'])
    with col2:
        st.metric("🚫 Заблокировано", st.session_state.sim_stats['blocked'], 
                 delta=f"{st.session_state.sim_stats['fraud_caught']} fraud")
    with col3:
        st.metric("✅ Одобрено", st.session_state.sim_stats['approved'])
    with col4:
        accuracy = 0
        if st.session_state.sim_stats['blocked'] > 0:
            accuracy = (st.session_state.sim_stats['fraud_caught'] / 
                       st.session_state.sim_stats['blocked'] * 100)
        st.metric("🎯 Precision", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    # Кнопки управления
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_button = st.button("▶️ ЗАПУСТИТЬ СИМУЛЯЦИЮ", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Сбросить статистику", use_container_width=True):
            st.session_state.sim_stats = {
                'total': 0, 'blocked': 0, 'approved': 0, 
                'reviewed': 0, 'fraud_caught': 0, 'fraud_missed': 0
            }
            st.rerun()
    
    st.markdown("---")
    
    # Запуск симуляции
    if start_button:
        st.subheader("🔄 Обработка входящих транзакций...")
        
        # Загрузить СЛУЧАЙНЫЕ транзакции
        transactions_to_check = load_random_transactions_for_simulation(10)
        
        if len(transactions_to_check) == 0:
            st.warning("📭 Нет транзакций в базе данных")
            return
        
        st.info(f"✅ Загружено {len(transactions_to_check)} случайных транзакций из базы")
        
        # Placeholder для динамического обновления
        progress_bar = st.progress(0)
        status_text = st.empty()
        transaction_card = st.empty()
        result_card = st.empty()
        
        for idx, trans in enumerate(transactions_to_check):
            # Обновить прогресс
            progress = (idx + 1) / len(transactions_to_check)
            progress_bar.progress(progress)
            status_text.markdown(f"**Транзакция {idx + 1} из {len(transactions_to_check)}**")
            
            # Показать карточку транзакции
            with transaction_card.container():
                st.markdown("### 📥 Входящая транзакция")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**ID:** {trans['transaction_id']}")
                    st.write(f"**Клиент:** {trans['customer_id'][:8]}...")
                
                with col2:
                    st.write(f"**💰 Сумма:** {trans['amount']:,.2f} ₸")
                    dt = pd.to_datetime(trans['transaction_datetime'])
                    st.write(f"**⏰ Время:** {dt.strftime('%H:%M:%S')}")
                
                with col3:
                    is_night = "🌙 Ночь" if 0 <= dt.hour < 6 else "☀️ День"
                    st.write(f"**Период:** {is_night}")
                    st.write(f"**📅 Дата:** {dt.strftime('%Y-%m-%d')}")
                
                with col4:
                    # Показать РЕАЛЬНЫЙ статус (для проверки)
                    if trans.get('is_fraud', False):
                        st.write("**🔴 РЕАЛЬНО:** Фрод")
                    else:
                        st.write("**🟢 РЕАЛЬНО:** Чисто")
            
            time.sleep(delay * 0.5)
            
            # Анализ с помощью ML
            with st.spinner("🤖 Анализ ML модели..."):
                time.sleep(delay * 0.5)
                
                # Получить поведенческие данные
                behavior = get_customer_behavior(
                    trans['customer_id'],
                    dt.date()
                )
                
                # Извлечь признаки
                features = extract_features_from_transaction(trans, behavior)
                
                # Вызвать ML API
                prediction = call_ml_api(features)
            
            if not prediction:
                st.error("❌ ML API недоступен")
                continue
            
            # Показать результат анализа
            probability = prediction['probability']
            risk_level = prediction.get('riskLevel', 'UNKNOWN')
            
            # Определить решение на основе порога
            if probability >= threshold:
                decision = "BLOCK"
                decision_color = "error"
                decision_emoji = "🚫"
                decision_text = "ЗАБЛОКИРОВАТЬ"
            elif probability >= 0.50:
                decision = "REVIEW"
                decision_color = "warning"
                decision_emoji = "⚠️"
                decision_text = "НА ПРОВЕРКУ"
            else:
                decision = "APPROVE"
                decision_color = "success"
                decision_emoji = "✅"
                decision_text = "ОДОБРИТЬ"
            
            with result_card.container():
                st.markdown("### 🎯 Результат анализа")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # ML решение
                    if decision == "BLOCK":
                        st.markdown(f"""
                        <div style='background-color: #fee2e2; border-left: 5px solid #ef4444; 
                                    padding: 20px; border-radius: 10px;'>
                            <h2 style='color: #dc2626; margin: 0;'>{decision_emoji} {decision_text}</h2>
                            <h1 style='color: #dc2626; margin: 10px 0;'>{probability*100:.1f}%</h1>
                            <p style='margin: 0;'><strong>ML Решение</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif decision == "REVIEW":
                        st.markdown(f"""
                        <div style='background-color: #fef3c7; border-left: 5px solid #f59e0b; 
                                    padding: 20px; border-radius: 10px;'>
                            <h2 style='color: #d97706; margin: 0;'>{decision_emoji} {decision_text}</h2>
                            <h1 style='color: #d97706; margin: 10px 0;'>{probability*100:.1f}%</h1>
                            <p style='margin: 0;'><strong>ML Решение</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: #d1fae5; border-left: 5px solid #10b981; 
                                    padding: 20px; border-radius: 10px;'>
                            <h2 style='color: #059669; margin: 0;'>{decision_emoji} {decision_text}</h2>
                            <h1 style='color: #059669; margin: 10px 0;'>{probability*100:.1f}%</h1>
                            <p style='margin: 0;'><strong>ML Решение</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Реальность
                    is_fraud_real = trans.get('is_fraud', False)
                    if is_fraud_real:
                        st.markdown("""
                        <div style='background-color: #fee2e2; border: 2px solid #ef4444; 
                                    padding: 20px; border-radius: 10px;'>
                            <h3 style='color: #dc2626; margin: 0;'>🔴 РЕАЛЬНОСТЬ</h3>
                            <h2 style='color: #dc2626; margin: 10px 0;'>ФРОД</h2>
                            <p style='margin: 0;'><strong>Настоящий статус</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #d1fae5; border: 2px solid #10b981; 
                                    padding: 20px; border-radius: 10px;'>
                            <h3 style='color: #059669; margin: 0;'>🟢 РЕАЛЬНОСТЬ</h3>
                            <h2 style='color: #059669; margin: 10px 0;'>ЧИСТО</h2>
                            <p style='margin: 0;'><strong>Настоящий статус</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Топ-3 фактора
                if 'topFactors' in prediction:
                    st.markdown("**🔍 Ключевые факторы:**")
                    for factor in prediction['topFactors'][:3]:
                        direction = "↑" if factor['impact'] > 0 else "↓"
                        st.markdown(f"- {direction} **{factor['featureName']}**: {factor['impact']:.3f}")
            
            time.sleep(delay)
            
            # Автоматически применить решение
            if auto_mode:
                success = update_transaction_status(trans['id'], decision.lower())
                
                # Обновить статистику
                st.session_state.sim_stats['total'] += 1
                
                is_fraud_real = trans.get('is_fraud', False)
                
                if decision == "BLOCK":
                    st.session_state.sim_stats['blocked'] += 1
                    if is_fraud_real:
                        st.session_state.sim_stats['fraud_caught'] += 1
                        st.success("✅ ВЕРНО! Фрод пойман!")
                    else:
                        st.warning("⚠️ FALSE POSITIVE (заблокировали чистую)")
                
                elif decision == "APPROVE":
                    st.session_state.sim_stats['approved'] += 1
                    if is_fraud_real:
                        st.session_state.sim_stats['fraud_missed'] += 1
                        st.error("❌ ПРОПУЩЕН ФРОД!")
                    else:
                        st.success("✅ ВЕРНО! Чистая пропущена")
                
                else:
                    st.session_state.sim_stats['reviewed'] += 1
                
                if success:
                    st.info(f"💾 Решение применено: {decision_text}")
                else:
                    st.error(f"❌ Ошибка: решение НЕ применено")
                
                time.sleep(delay * 0.5)
        
        st.balloons()
        st.success(f"🎉 Симуляция завершена! Обработано {len(transactions_to_check)} транзакций")
        
        # Итоговая статистика
        st.markdown("---")
        st.subheader("📊 Итоги симуляции")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Всего", st.session_state.sim_stats['total'])
            st.metric("Заблокировано", st.session_state.sim_stats['blocked'])
        
        with col2:
            st.metric("Фродов поймано", st.session_state.sim_stats['fraud_caught'])
            st.metric("Фродов пропущено", st.session_state.sim_stats['fraud_missed'])
        
        with col3:
            if st.session_state.sim_stats['blocked'] > 0:
                precision = (st.session_state.sim_stats['fraud_caught'] / 
                           st.session_state.sim_stats['blocked'] * 100)
                st.metric("Precision", f"{precision:.1f}%")
            
            if st.session_state.sim_stats['fraud_caught'] + st.session_state.sim_stats['fraud_missed'] > 0:
                recall = (st.session_state.sim_stats['fraud_caught'] / 
                         (st.session_state.sim_stats['fraud_caught'] + 
                          st.session_state.sim_stats['fraud_missed']) * 100)
                st.metric("Recall", f"{recall:.1f}%")

def main():
    st.markdown('<h1 class="main-header">🛡️ ML Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time транзакционный мониторинг с использованием XGBoost + SHAP**")
    st.markdown("---")
    
    # Sidebar навигация
    st.sidebar.title("🎯 Навигация")
    page = st.sidebar.radio(
        "Выберите страницу:",
        ["📊 Dashboard", 
         "🔍 Анализатор транзакций", 
         "🎮 Симулятор блокировки",  # ← НОВАЯ СТРАНИЦА
         "📈 Статистика", 
         "⚙️ Настройки"]
    )
    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🔍 Анализатор транзакций":
        show_transaction_analyzer()
    elif page == "🎮 Симулятор блокировки":  # ← НОВЫЙ ОБРАБОТЧИК
        show_simulator()
    elif page == "📈 Статистика":
        show_statistics()
    elif page == "⚙️ Настройки":
        show_settings()

# ==================== СТРАНИЦА: DASHBOARD ====================

def show_dashboard():
    st.header("📊 Главная панель")
    
    # Загрузить статистику
    stats = load_statistics()
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔢 Всего транзакций",
            value=f"{stats['total_transactions']:,}",
            delta=None
        )
    
    with col2:
        fraud_rate = (stats['fraud_count'] / stats['total_transactions'] * 100) if stats['total_transactions'] > 0 else 0
        st.metric(
            label="⚠️ Мошеннических",
            value=f"{stats['fraud_count']:,}",
            delta=f"{fraud_rate:.1f}%"
        )
    
    with col3:
        st.metric(
            label="💰 Сохранено средств",
            value=f"{stats['fraud_amount_saved']:,.0f} ₸",
            delta="Заблокировано"
        )
    
    with col4:
        st.metric(
            label="👥 Уникальных клиентов",
            value=f"{stats['unique_customers']:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Графики
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение по часам
        hourly_stats = load_hourly_stats()
        fig = px.line(
            hourly_stats,
            x='hour',
            y=['total', 'fraud'],
            title="📈 Распределение транзакций по часам",
            labels={'value': 'Количество', 'hour': 'Час'},
            color_discrete_map={'total': '#667eea', 'fraud': '#ef4444'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart - распределение
        fraud_dist = pd.DataFrame({
            'Категория': ['Чистые', 'Мошеннические'],
            'Количество': [stats['clean_count'], stats['fraud_count']]
        })
        fig = px.pie(
            fraud_dist,
            values='Количество',
            names='Категория',
            title="🥧 Распределение транзакций",
            color='Категория',
            color_discrete_map={'Чистые': '#10b981', 'Мошеннические': '#ef4444'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Последние транзакции
    st.subheader("📋 Последние транзакции")
    df = load_transactions(20)
    
    if not df.empty:
        # Форматировать для отображения
        df_display = df.copy()
        df_display['amount'] = df_display['amount'].apply(lambda x: f"{x:,.2f} ₸")
        df_display['fraud_probability'] = df_display['fraud_probability'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
        df_display['is_fraud'] = df_display['is_fraud'].apply(
            lambda x: "🔴 Fraud" if x else "🟢 Clean"
        )
        
        st.dataframe(
            df_display[['transaction_id', 'customer_id', 'amount', 'fraud_probability', 'is_fraud', 'status']],
            use_container_width=True,
            height=400
        )
    else:
        st.info("📭 Нет транзакций для отображения")

# ==================== СТРАНИЦА: АНАЛИЗАТОР ====================

def show_transaction_analyzer():
    st.header("🔍 Анализатор транзакций")
    st.markdown("Выберите транзакцию и проверьте её с помощью ML модели")
    
    # Загрузить транзакции
    df = load_transactions(500)
    
    if df.empty:
        st.warning("📭 Нет транзакций в базе данных")
        return
    
    # Выбор транзакции
    st.subheader("1️⃣ Выберите транзакцию для анализа")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Фильтры
        filter_type = st.selectbox(
            "Фильтр:",
            ["Все транзакции", "Только подозрительные", "Только чистые", "Без анализа"]
        )
        
        if filter_type == "Только подозрительные":
            df_filtered = df[df['is_fraud'] == True]
        elif filter_type == "Только чистые":
            df_filtered = df[df['is_fraud'] == False]
        elif filter_type == "Без анализа":
            df_filtered = df[df['fraud_probability'].isna()]
        else:
            df_filtered = df
    
    with col2:
        # Сортировка
        sort_by = st.selectbox("Сортировать по:", ["Дате (новые)", "Сумме (большие)", "Сумме (малые)"])
        
        if sort_by == "Сумме (большие)":
            df_filtered = df_filtered.sort_values('amount', ascending=False)
        elif sort_by == "Сумме (малые)":
            df_filtered = df_filtered.sort_values('amount', ascending=True)
    
    # Показать список транзакций
    st.markdown("### Выберите транзакцию:")
    
    if df_filtered.empty:
        st.info("Нет транзакций по выбранному фильтру")
        return
    
    # Отобразить красиво
    for idx, row in df_filtered.head(10).iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.write(f"**{row['transaction_id']}**")
            st.caption(f"Customer: {row['customer_id']}")
        
        with col2:
            st.write(f"💰 **{row['amount']:,.2f} ₸**")
            st.caption(f"{row['transaction_datetime']}")
        
        with col3:
            if pd.notna(row['fraud_probability']):
                prob = row['fraud_probability'] * 100
                color = "🔴" if prob >= 70 else "🟡" if prob >= 50 else "🟢"
                st.write(f"{color} **{prob:.1f}%**")
                st.caption(f"Status: {row['status']}")
            else:
                st.write("⚪ **Не проверено**")
                st.caption("Требует анализа")
        
        with col4:
            if st.button("Анализ", key=f"analyze_{row['id']}"):
                st.session_state['selected_transaction_id'] = row['id']
                st.rerun()
        
        st.markdown("---")
    
    # Если выбрана транзакция
    if 'selected_transaction_id' in st.session_state:
        st.markdown("---")
        analyze_selected_transaction(st.session_state['selected_transaction_id'])

def analyze_selected_transaction(transaction_id):
    """Анализ выбранной транзакции"""
    st.subheader("2️⃣ Анализ транзакции")
    
    # Загрузить транзакцию
    trans = get_transaction_by_id(transaction_id)
    
    if not trans:
        st.error("Транзакция не найдена")
        return
    
    # Показать детали
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Детали транзакции:**")
        st.write(f"**ID:** {trans['transaction_id']}")
        st.write(f"**Сумма:** {trans['amount']:,.2f} ₸")
        st.write(f"**Клиент:** {trans['customer_id']}")
        st.write(f"**Получатель:** {trans['recipient_id']}")
    
    with col2:
        st.markdown("**⏰ Время:**")
        dt = trans['transaction_datetime']
        st.write(f"**Дата:** {dt.date()}")
        st.write(f"**Время:** {dt.time()}")
        st.write(f"**День недели:** {dt.strftime('%A')}")
        is_night = "🌙 Да" if 0 <= dt.hour < 6 else "☀️ Нет"
        st.write(f"**Ночь:** {is_night}")
    
    with col3:
        st.markdown("**📊 Статус:**")
        if pd.notna(trans['fraud_probability']):
            prob = trans['fraud_probability'] * 100
            st.write(f"**Вероятность:** {prob:.1f}%")
            st.write(f"**Мошенничество:** {'🔴 Да' if trans['is_fraud'] else '🟢 Нет'}")
            st.write(f"**Статус:** {trans['status']}")
        else:
            st.write("⚪ **Не проанализировано**")
    
    st.markdown("---")
    
    # Кнопка анализа
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🤖 Запустить ML анализ", use_container_width=True, type="primary"):
            with st.spinner("🔄 Анализ в процессе..."):
                # Получить поведенческие паттерны
                behavior = get_customer_behavior(
                    trans['customer_id'],
                    trans['transaction_datetime'].date()
                )
                
                # Извлечь признаки
                features = extract_features_from_transaction(trans, behavior)
                
                # Вызвать ML API
                prediction = call_ml_api(features)
                
                if prediction:
                    st.session_state['prediction'] = prediction
                    st.session_state['features'] = features
                    st.rerun()
    
    # Показать результаты если есть
    if 'prediction' in st.session_state:
        st.markdown("---")
        show_prediction_results(trans, st.session_state['prediction'], st.session_state['features'])

def show_prediction_results(trans, prediction, features):
    """Показать результаты предсказания С AI РЕКОМЕНДАЦИЕЙ"""
    st.subheader("3️⃣ Результаты ML анализа")
    
    probability = prediction['probability']
    is_fraud = prediction['isFraud']
    risk_level = prediction.get('riskLevel', 'unknown')
    action = prediction.get('action', 'REVIEW')
    
    # Большая карточка с результатом (как было)
    if probability >= 0.85:
        st.markdown(f"""
        <div class="fraud-card">
            <h2>🚨 ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА</h2>
            <h1 style='color: #dc2626;'>{probability*100:.1f}%</h1>
            <p><strong>Рекомендация:</strong> Заблокировать транзакцию</p>
        </div>
        """, unsafe_allow_html=True)
    elif probability >= 0.50:
        st.markdown(f"""
        <div class="review-card">
            <h2>⚠️ СРЕДНИЙ РИСК</h2>
            <h1 style='color: #d97706;'>{probability*100:.1f}%</h1>
            <p><strong>Рекомендация:</strong> Требуется проверка</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="clean-card">
            <h2>✅ НИЗКИЙ РИСК</h2>
            <h1 style='color: #059669;'>{probability*100:.1f}%</h1>
            <p><strong>Рекомендация:</strong> Одобрить транзакцию</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 🤖 AI РЕКОМЕНДАЦИЯ (НОВОЕ!)
    with st.spinner("🤖 Генерация AI рекомендации..."):
        ai_recommendation = ai_service.generate_recommendation(
            trans,
            prediction,
            prediction.get('topFactors', [])
        )
    
    st.markdown("### 🤖 AI Рекомендация для аналитика")
    st.info(ai_recommendation)
    
    st.markdown("---")
    
    # Топ факторы риска (как было)
    st.subheader("🎯 Ключевые факторы риска (SHAP values)")
    
    if 'topFactors' in prediction:
        top_factors = prediction['topFactors'][:5]
        
        for factor in top_factors:
            feature_name = factor['feature']
            impact = factor['impact']
            direction = "↑ УВЕЛИЧИВАЕТ" if impact > 0 else "↓ СНИЖАЕТ"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                abs_impact = abs(impact)
                st.markdown(f"**{factor.get('featureName', feature_name)}**")
                st.progress(min(abs_impact, 1.0))
            
            with col2:
                st.markdown(f"**{direction}**")
                st.caption(f"{abs_impact:.3f}")
            
            st.markdown("---")
    
    # Принятие решения (как было)
    st.subheader("4️⃣ Принятие решения")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ ОДОБРИТЬ", use_container_width=True, type="secondary"):
            update_transaction_status(trans['id'], 'approved')
            st.success("✅ Транзакция одобрена!")
            st.balloons()
            if 'prediction' in st.session_state:
                del st.session_state['prediction']
            if 'selected_transaction_id' in st.session_state:
                del st.session_state['selected_transaction_id']
            st.rerun()
    
    with col2:
        if st.button("⚠️ НА ПРОВЕРКУ", use_container_width=True):
            update_transaction_status(trans['id'], 'review')
            st.warning("⚠️ Отправлено на проверку")
            if 'prediction' in st.session_state:
                del st.session_state['prediction']
            if 'selected_transaction_id' in st.session_state:
                del st.session_state['selected_transaction_id']
            st.rerun()
    
    with col3:
        if st.button("🚫 ЗАБЛОКИРОВАТЬ", use_container_width=True, type="primary"):
            update_transaction_status(trans['id'], 'blocked')
            st.error("🚫 Транзакция заблокирована!")
            if 'prediction' in st.session_state:
                del st.session_state['prediction']
            if 'selected_transaction_id' in st.session_state:
                del st.session_state['selected_transaction_id']
            st.rerun()
    
    with col4:
        if st.button("🔄 Новый анализ", use_container_width=True):
            if 'prediction' in st.session_state:
                del st.session_state['prediction']
            if 'selected_transaction_id' in st.session_state:
                del st.session_state['selected_transaction_id']
            st.rerun()

# ==================== СТРАНИЦА: СТАТИСТИКА ====================

def show_statistics():
    st.header("📈 Детальная статистика")
    
    stats = load_statistics()
    
    st.subheader("📊 Общие показатели")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Всего транзакций", f"{stats['total_transactions']:,}")
        st.metric("Средняя сумма", f"{stats['avg_amount']:,.2f} ₸")
        st.metric("Общая сумма", f"{stats['total_amount']:,.2f} ₸")
    
    with col2:
        st.metric("Мошеннических", stats['fraud_count'])
        st.metric("Чистых", stats['clean_count'])
        fraud_rate = (stats['fraud_count'] / stats['total_transactions'] * 100) if stats['total_transactions'] > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    st.markdown("---")
    
    # Дополнительные графики
    st.subheader("📉 Дополнительная аналитика")
    
    df = load_transactions(1000)
    
    if not df.empty:
        # График распределения сумм
        fig = px.histogram(
            df,
            x='amount',
            color='is_fraud',
            title="Распределение сумм транзакций",
            nbins=50,
            color_discrete_map={True: '#ef4444', False: '#10b981'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== СТРАНИЦА: НАСТРОЙКИ ====================

def show_settings():
    st.header("⚙️ Настройки системы")

    # НОВЫЙ РАЗДЕЛ - настройки AI
    st.markdown("---")
    st.subheader("🤖 Настройки AI")
    
    # Поле для API ключа
    current_key = OPENAI_API_KEY or ""
    masked_key = current_key[:7] + "..." if current_key else "Не установлен"
    
    st.info(f"**Текущий ключ:** {masked_key}")
    
    new_api_key = st.text_input(
        "OpenAI API Key:",
        value="",
        type="password",
        placeholder="sk-...",
        help="Введите ваш OpenAI API ключ для активации AI рекомендаций"
    )
    
    if st.button("💾 Сохранить API ключ"):
        if new_api_key and new_api_key.startswith('sk-'):
            # Сохранить в secrets.toml
            secrets_path = Path(".streamlit/secrets.toml")
            secrets_path.parent.mkdir(exist_ok=True)
            
            with open(secrets_path, 'w') as f:
                f.write(f'openai_api_key = "{new_api_key}"\n')
            
            st.success("✅ API ключ сохранен! Перезапустите Dashboard.")
        else:
            st.error("❌ Неверный формат ключа (должен начинаться с 'sk-')")
    
    st.markdown("""
    **Как получить API ключ:**
    1. Зайдите на https://platform.openai.com/api-keys
    2. Создайте новый API ключ
    3. Скопируйте и вставьте сюда
    4. Сохраните и перезапустите Dashboard
    """)
    
    if ai_service.enabled:
        st.success("✅ AI рекомендации АКТИВНЫ")
    else:
        st.warning("⚠️ AI рекомендации ОТКЛЮЧЕНЫ (работает fallback режим)")
    
    st.subheader("🔧 Конфигурация ML API")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("ML API URL", value=ML_API_URL, disabled=True)
        
        # Проверка подключения
        if st.button("🔍 Проверить подключение"):
            try:
                response = requests.get(f"{ML_API_URL}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ ML API доступен!")
                    st.json(data)
                else:
                    st.error(f"❌ ML API недоступен (код: {response.status_code})")
            except Exception as e:
                st.error(f"❌ Ошибка подключения: {e}")
                st.info("💡 Запустите ML сервис: `python ml_service.py`")
    
    with col2:
        st.text_input("PostgreSQL Host", value=DB_CONFIG['host'], disabled=True)
        st.text_input("Database", value=DB_CONFIG['database'], disabled=True)
        
        # Проверка БД
        if st.button("🗄️ Проверить БД"):
            try:
                conn = get_db_connection()
                st.success("✅ База данных доступна!")
                
                # Показать статистику таблиц
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transactions")
                trans_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM customer_behavior_patterns")
                behavior_count = cursor.fetchone()[0]
                
                st.info(f"📊 Транзакций: {trans_count}")
                st.info(f"📊 Паттернов поведения: {behavior_count}")
                
                conn.close()
            except Exception as e:
                st.error(f"❌ Ошибка подключения к БД: {e}")
    
    st.markdown("---")
    
    st.subheader("🎨 Настройки отображения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transactions_limit = st.slider(
            "Количество транзакций на странице",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
    
    with col2:
        refresh_interval = st.slider(
            "Интервал обновления (секунды)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    
    st.markdown("---")
    
    st.subheader("📚 Информация о модели")
    
    try:
        response = requests.get(f"{ML_API_URL}/feature_importance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            importance = data.get('importance', {})
            
            if importance:
                st.write("**Топ-10 важных признаков:**")
                
                df_importance = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in list(importance.items())[:10]
                ])
                
                fig = px.bar(
                    df_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance (XGBoost Gain)",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Не удалось загрузить Feature Importance: {e}")
    
    st.markdown("---")
    
    st.subheader("🧹 Управление данными")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Очистить кэш", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Кэш очищен!")
    
    with col2:
        if st.button("📥 Экспорт данных", use_container_width=True):
            df = load_transactions(1000)
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Скачать CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📊 Генерировать отчет", use_container_width=True):
            st.info("🚧 Функция в разработке")

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    main()