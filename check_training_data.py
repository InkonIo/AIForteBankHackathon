"""
Проверка данных в таблицах для обучения модели
"""

import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_db',
    'user': 'postgres',
    'password': 'Alikhancool20!'
}

def check_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    print("="*70)
    print("📊 ПРОВЕРКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
    print("="*70)
    
    # 1. Проверка transactions
    print("\n1️⃣  ТАБЛИЦА TRANSACTIONS:")
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_trans = cursor.fetchone()[0]
    print(f"   Всего транзакций: {total_trans}")
    
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")
    fraud_trans = cursor.fetchone()[0]
    print(f"   Мошеннических: {fraud_trans}")
    
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = false")
    clean_trans = cursor.fetchone()[0]
    print(f"   Чистых: {clean_trans}")
    
    # 2. Проверка customer_behavior_patterns
    print("\n2️⃣  ТАБЛИЦА CUSTOMER_BEHAVIOR_PATTERNS:")
    cursor.execute("SELECT COUNT(*) FROM customer_behavior_patterns")
    total_patterns = cursor.fetchone()[0]
    print(f"   Всего записей: {total_patterns}")
    
    cursor.execute("SELECT COUNT(DISTINCT customer_id) FROM customer_behavior_patterns")
    unique_customers = cursor.fetchone()[0]
    print(f"   Уникальных клиентов: {unique_customers}")
    
    # 3. Проверка JOIN
    print("\n3️⃣  ПРОВЕРКА СОЕДИНЕНИЯ ТАБЛИЦ:")
    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions t
        LEFT JOIN customer_behavior_patterns cb 
            ON t.customer_id = cb.customer_id 
            AND DATE(t.transaction_datetime) = cb.trans_date
        WHERE t.transaction_datetime >= NOW() - INTERVAL '90 days'
    """)
    joined_count = cursor.fetchone()[0]
    print(f"   Транзакций с паттернами (90 дней): {joined_count}")
    
    # 4. Проверка мошеннических с паттернами
    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions t
        LEFT JOIN customer_behavior_patterns cb 
            ON t.customer_id = cb.customer_id 
            AND DATE(t.transaction_datetime) = cb.trans_date
        WHERE t.is_fraud = true 
        AND t.transaction_datetime >= NOW() - INTERVAL '90 days'
    """)
    fraud_with_patterns = cursor.fetchone()[0]
    print(f"   Мошеннических транзакций: {fraud_with_patterns}")
    
    # 5. Рекомендации
    print("\n" + "="*70)
    print("📋 ОЦЕНКА ГОТОВНОСТИ К ОБУЧЕНИЮ:")
    print("="*70)
    
    if total_trans < 100:
        print("❌ Недостаточно транзакций (нужно минимум 100)")
    else:
        print(f"✅ Транзакций достаточно: {total_trans}")
    
    if fraud_trans < 10:
        print("❌ Недостаточно мошеннических примеров (нужно минимум 10)")
    else:
        print(f"✅ Мошеннических транзакций достаточно: {fraud_trans}")
    
    if joined_count < 100:
        print("⚠️  Мало транзакций с поведенческими паттернами")
    else:
        print(f"✅ Данных для обучения достаточно: {joined_count}")
    
    if total_trans >= 100 and fraud_trans >= 10:
        print("\n🎯 МОЖНО ЗАПУСКАТЬ ОБУЧЕНИЕ МОДЕЛИ!")
        print("   Команда: python train_improved_model.py")
    else:
        print("\n⚠️  НУЖНО БОЛЬШЕ ДАННЫХ!")
        print("   Добавьте больше транзакций или запустите генератор данных")
    
    conn.close()

if __name__ == "__main__":
    check_data()