// ==================== CONFIGURATION ====================

const API_URL = 'http://localhost:5001';

// Текущая транзакция (для сравнения с реальной меткой)
let currentTransaction = null;

// ==================== UTILITY FUNCTIONS ====================

function formatCurrency(amount) {
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'KZT',
        minimumFractionDigits: 0
    }).format(amount);
}

function formatDate(date) {
    return new Date(date).toLocaleString('ru-RU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// ==================== API CALLS ====================

async function checkTransaction(formData) {
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            customer_id: formData.customerId,
            amount: parseFloat(formData.amount),
            datetime: formData.datetime || new Date().toISOString(),
            recipient_id: formData.recipientId || 'REC_' + Math.random().toString(36).substr(2, 9)
        })
    });
    
    if (!response.ok) {
        throw new Error('Ошибка API');
    }
    
    return await response.json();
}

async function loadStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        const stats = await response.json();
        
        // Обновляем статистику
        document.getElementById('totalTransactions').textContent = 
            stats.total_transactions.toLocaleString();
        document.getElementById('fraudDetected').textContent = 
            stats.fraud_detected.toLocaleString();
        document.getElementById('accuracyRate').textContent = 
            (stats.recall * 100).toFixed(1) + '%';
    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
    }
}

async function loadSampleTransaction(type) {
    try {
        showLoading();
        
        const response = await fetch(`${API_URL}/get_sample_transaction?type=${type}`);
        
        if (!response.ok) {
            throw new Error('Не удалось загрузить транзакцию');
        }
        
        const transaction = await response.json();
        currentTransaction = transaction;
        
        // Заполняем форму
        document.getElementById('customerId').value = transaction.customer_id;
        document.getElementById('amount').value = transaction.amount;
        document.getElementById('recipientId').value = transaction.recipient_id || '';
        
        // Отправляем на проверку
        const result = await checkTransaction({
            customerId: transaction.customer_id,
            amount: transaction.amount,
            recipientId: transaction.recipient_id,
            datetime: transaction.datetime
        });
        
        // Показываем результат с сравнением
        showResult(result, {
            customerId: transaction.customer_id,
            amount: transaction.amount,
            recipientId: transaction.recipient_id,
            datetime: transaction.datetime
        });
        
    } catch (error) {
        showError(error.message);
    }
}

// Делаем функцию глобальной для onclick
window.loadSampleTransaction = loadSampleTransaction;

// ==================== FORM HANDLER ====================

document.getElementById('transactionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        customerId: document.getElementById('customerId').value,
        amount: document.getElementById('amount').value,
        recipientId: document.getElementById('recipientId').value
    };
    
    showLoading();
    
    try {
        const result = await checkTransaction(formData);
        currentTransaction = null; // Нет реальной метки для ручного ввода
        showResult(result, formData);
    } catch (error) {
        showError(error.message);
    }
});

// ==================== UI RENDERING ====================

function showLoading() {
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.innerHTML = `
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <h3>Анализ транзакции...</h3>
            <p>ML-модель обрабатывает 54 признака</p>
            
            <!-- Анимированный процесс работы модели -->
            <div class="ml-process-flow">
                <div class="process-stage" id="stage1">
                    <div class="stage-icon">
                        <i class="fas fa-download"></i>
                    </div>
                    <div class="stage-content">
                        <div class="stage-title">Загрузка данных</div>
                        <div class="stage-description">Получение информации о транзакции и клиенте из базы данных</div>
                    </div>
                    <div class="stage-status">⏳ Ожидание</div>
                </div>
                
                <div class="stage-arrow">
                    <i class="fas fa-arrow-down"></i>
                </div>
                
                <div class="process-stage" id="stage2">
                    <div class="stage-icon">
                        <i class="fas fa-wrench"></i>
                    </div>
                    <div class="stage-content">
                        <div class="stage-title">Извлечение признаков</div>
                        <div class="stage-description">Создание 54 признаков: сумма, время, поведение клиента</div>
                    </div>
                    <div class="stage-status">⏳ Ожидание</div>
                </div>
                
                <div class="stage-arrow">
                    <i class="fas fa-arrow-down"></i>
                </div>
                
                <div class="process-stage" id="stage3">
                    <div class="stage-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="stage-content">
                        <div class="stage-title">ML предсказание</div>
                        <div class="stage-description">LightGBM модель анализирует паттерны и вычисляет вероятность</div>
                    </div>
                    <div class="stage-status">⏳ Ожидание</div>
                </div>
                
                <div class="stage-arrow">
                    <i class="fas fa-arrow-down"></i>
                </div>
                
                <div class="process-stage" id="stage4">
                    <div class="stage-icon">
                        <i class="fas fa-gavel"></i>
                    </div>
                    <div class="stage-content">
                        <div class="stage-title">Принятие решения</div>
                        <div class="stage-description">Определение действия: одобрить, проверить или заблокировать</div>
                    </div>
                    <div class="stage-status">⏳ Ожидание</div>
                </div>
            </div>
        </div>
    `;
    
    // Запускаем анимацию этапов
    animateProcessStages();
}

function animateProcessStages() {
    // УВЕЛИЧИЛ ВРЕМЯ! Теперь каждый этап длится дольше
    const stages = [
        { id: 'stage1', delay: 1500, duration: 1500 },   // Загрузка данных
        { id: 'stage2', delay: 3500, duration: 1500 },  // Извлечение признаков
        { id: 'stage3', delay: 5500, duration: 2000 },  // ML предсказание (дольше всех)
        { id: 'stage4', delay: 8000, duration: 1500 }    // Принятие решения
    ];
    
    stages.forEach(stage => {
        setTimeout(() => {
            const element = document.getElementById(stage.id);
            if (element) {
                element.classList.add('active');
                const status = element.querySelector('.stage-status');
                status.textContent = '🔄 Обработка...';
                
                // Используем duration из настроек
                setTimeout(() => {
                    element.classList.remove('active');
                    element.classList.add('completed');
                    status.textContent = '✅ Готово';
                }, stage.duration);
            }
        }, stage.delay);
    });
}

function showResult(result, formData) {
    const resultPanel = document.getElementById('resultPanel');
    
    const probability = result.fraud_probability;
    const decision = result.decision.toLowerCase();
    
    // Проверяем, есть ли реальная метка для сравнения
    const hasActualLabel = currentTransaction && currentTransaction.is_fraud !== undefined;
    const isCorrect = hasActualLabel ? checkPredictionCorrectness(probability) : null;
    
    resultPanel.innerHTML = `
        <div class="result-state">
            ${hasActualLabel ? renderComparisonBanner(isCorrect) : ''}
            
            <!-- Risk Gauge -->
            <div class="risk-gauge-container">
                <svg class="risk-gauge" viewBox="0 0 200 120">
                    <path class="gauge-bg"
                          d="M 20 100 A 80 80 0 0 1 180 100"
                          stroke-linecap="round"/>
                    <path class="gauge-fill"
                          d="M 20 100 A 80 80 0 0 1 180 100"
                          stroke="${getGaugeColor(probability)}"
                          stroke-dasharray="251.2"
                          stroke-dashoffset="${251.2 - (probability / 100) * 251.2}"
                          stroke-linecap="round"/>
                    <circle cx="100" cy="100" r="3" fill="#333"/>
                    <line class="gauge-needle"
                          x1="100" y1="100"
                          x2="100" y2="35"
                          stroke="#333"
                          stroke-width="3"
                          transform="rotate(${-90 + (probability / 100) * 180}, 100, 100)"/>
                </svg>
                <span class="risk-percentage" style="color: ${getGaugeColor(probability)}">
                    ${probability.toFixed(1)}%
                </span>
                <span class="risk-label">Вероятность мошенничества</span>
            </div>
            
            <div class="decision-badge ${decision}">
                <i class="fas ${getDecisionIcon(decision)}"></i>
                <span>${result.decision_label}</span>
            </div>
            
            <!-- Explanation -->
            <div class="transaction-details">
                <h4><i class="fas fa-lightbulb"></i> Объяснение решения</h4>
                <p style="color: var(--gray-700); line-height: 1.8; margin-top: 0.5rem;">
                    ${result.decision_reason}
                </p>
            </div>
            
            <!-- Transaction Details -->
            <div class="transaction-details">
                <h4><i class="fas fa-file-invoice-dollar"></i> Детали транзакции</h4>
                ${hasActualLabel ? `
                    <div class="detail-row">
                        <span class="detail-label">ID транзакции:</span>
                        <span class="detail-value">${currentTransaction.transaction_id}</span>
                    </div>
                ` : ''}
                <div class="detail-row">
                    <span class="detail-label">Клиент:</span>
                    <span class="detail-value">${formData.customerId}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Сумма:</span>
                    <span class="detail-value">${formatCurrency(formData.amount)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Получатель:</span>
                    <span class="detail-value">${formData.recipientId || 'Не указан'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Время:</span>
                    <span class="detail-value">${formatDate(formData.datetime || new Date())}</span>
                </div>
                ${hasActualLabel ? `
                    <div class="detail-row">
                        <span class="detail-label">Реальная метка:</span>
                        <span class="detail-value" style="font-weight: 700; color: ${currentTransaction.is_fraud ? 'var(--danger)' : 'var(--success)'}">
                            ${currentTransaction.actual_label}
                        </span>
                    </div>
                ` : ''}
            </div>
            
            <!-- Feature Importance -->
            ${renderFeatureImportance(result.feature_importance)}
            
            <!-- Action Button -->
            <button class="btn btn-primary" onclick="resetForm()" style="margin-top: 2rem;">
                <i class="fas fa-plus"></i>
                <span>Проверить другую транзакцию</span>
            </button>
        </div>
    `;
}

function renderComparisonBanner(isCorrect) {
    if (isCorrect === null) return '';
    
    return `
        <div class="comparison-banner ${isCorrect ? 'correct' : 'incorrect'}">
            <div class="comparison-header">
                <i class="fas fa-${isCorrect ? 'check-circle' : 'times-circle'}"></i>
                <h4>${isCorrect ? '✅ Модель правильно классифицировала!' : '❌ Модель ошиблась'}</h4>
            </div>
            <div class="comparison-details">
                <strong>Реальная метка:</strong> ${currentTransaction.actual_label}<br>
                <strong>Предсказание модели:</strong> ${getPredictionLabel()}
            </div>
        </div>
    `;
}

function checkPredictionCorrectness(probability) {
    if (!currentTransaction) return null;
    
    const actualFraud = currentTransaction.is_fraud;
    const predictedFraud = probability > 50; // Порог 50%
    
    return actualFraud === predictedFraud;
}

function getPredictionLabel() {
    const resultPanel = document.getElementById('resultPanel');
    const riskText = resultPanel.querySelector('.risk-percentage');
    if (!riskText) return '';
    
    const prob = parseFloat(riskText.textContent);
    if (prob < 30) return 'Чистая (низкий риск)';
    if (prob < 50) return 'Требует проверки';
    if (prob < 70) return 'Подозрительная';
    return 'Мошенничество';
}

function renderFeatureImportance(importance) {
    if (!importance || importance.length === 0) return '';
    
    const maxImportance = Math.max(...importance.map(f => f.importance));
    
    return `
        <div class="feature-importance">
            <h4><i class="fas fa-chart-bar"></i> Важность признаков</h4>
            ${importance.slice(0, 6).map(feature => {
                const percentage = (feature.importance / maxImportance) * 100;
                return `
                    <div class="feature-bar">
                        <div class="feature-header">
                            <span class="feature-name">${translateFeatureName(feature.feature)}</span>
                            <span class="feature-value">${feature.importance}</span>
                        </div>
                        <div class="feature-bar-container">
                            <div class="feature-bar-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function showError(message) {
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon" style="color: var(--danger);">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <h3 style="color: var(--danger);">Ошибка</h3>
            <p>${message}</p>
            <p style="margin-top: 1rem; font-size: 0.9rem;">
                Убедитесь, что ML API запущен на http://localhost:5001
            </p>
            <button class="btn btn-primary" onclick="resetForm()" style="margin-top: 1.5rem; width: auto;">
                Попробовать снова
            </button>
        </div>
    `;
}

function resetForm() {
    document.getElementById('transactionForm').reset();
    currentTransaction = null;
    
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">
                <i class="fas fa-arrow-left"></i>
            </div>
            <h3>Ожидание данных</h3>
            <p>Заполните форму слева или загрузите пример из базы данных</p>
        </div>
    `;
}

window.resetForm = resetForm;

// ==================== HELPER FUNCTIONS ====================

function getGaugeColor(percentage) {
    if (percentage < 30) return '#10b981';
    if (percentage < 50) return '#f59e0b';
    if (percentage < 70) return '#ea580c';
    return '#ef4444';
}

function getDecisionIcon(decision) {
    const icons = {
        'approve': 'fa-check-circle',
        'verify': 'fa-magnifying-glass',
        'review': 'fa-triangle-exclamation',
        'block': 'fa-ban'
    };
    return icons[decision] || 'fa-question-circle';
}

function translateFeatureName(name) {
    const translations = {
        'amount': 'Сумма транзакции',
        'amount_log': 'Логарифм суммы',
        'amount_sqrt': 'Корень суммы',
        'is_night': 'Ночное время',
        'is_weekend': 'Выходной день',
        'is_peak_fraud_hour': 'Пик мошенничества',
        'burstiness_score': 'Показатель активности',
        'interval_zscore': 'Отклонение интервала',
        'logins_last_7_days': 'Входы за 7 дней',
        'logins_last_30_days': 'Входы за 30 дней',
        'avg_session_interval_sec': 'Средний интервал сессий',
        'hour': 'Час',
        'day_of_week': 'День недели',
        'hour_sin': 'Час (циклический)',
        'hour_cos': 'Час (косинус)',
        'login_ratio_7d_30d': 'Соотношение входов 7д/30д',
        'has_behavior_data': 'Есть данные поведения',
        'is_zero_activity': 'Нулевая активность',
        'device_diversity': 'Разнообразие устройств'
    };
    return translations[name] || name;
}

function explainFeature(name) {
    const explanations = {
        'amount': 'Размер перевода - большие суммы более подозрительны',
        'amount_log': 'Логарифмическая шкала суммы для лучшего анализа',
        'is_night': 'Мошенники часто действуют ночью (23:00-6:00)',
        'is_weekend': 'Активность в выходные может быть подозрительной',
        'is_peak_fraud_hour': 'Часы пик мошенничества (2:00-5:00 утра)',
        'burstiness_score': 'Как часто клиент заходит в приложение - резкие всплески подозрительны',
        'interval_zscore': 'Насколько необычен интервал между входами для этого клиента',
        'logins_last_7_days': 'Сколько раз клиент заходил за неделю',
        'logins_last_30_days': 'Сколько раз клиент заходил за месяц',
        'avg_session_interval_sec': 'Среднее время между входами клиента',
        'hour': 'Час совершения транзакции',
        'day_of_week': 'День недели транзакции',
        'login_ratio_7d_30d': 'Соотношение активности: недавняя vs общая',
        'has_behavior_data': 'Есть ли история поведения клиента',
        'is_zero_activity': 'Нет активности - очень подозрительно',
        'device_diversity': 'Использование разных устройств может быть признаком взлома'
    };
    return explanations[name] || 'Влияет на вероятность мошенничества';
}

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Forte Bank Fraud Detection System загружен');
    
    // Загружаем статистику
    loadStats();
    
    // Проверка доступности API
    fetch(`${API_URL}/health`)
        .then(response => response.json())
        .then(data => {
            console.log('✅ ML API доступен:', data);
        })
        .catch(error => {
            console.error('❌ ML API недоступен:', error);
            alert('⚠️ ML API сервис недоступен.\n\nЗапустите: python ml_api_service.py');
        });
    
    // Анимация метрик при загрузке
    setTimeout(() => {
        document.querySelectorAll('.metric-bar-fill').forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0';
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 500);
});
