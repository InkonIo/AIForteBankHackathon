// ==================== КОНФИГУРАЦИЯ ====================

const API_URL = 'http://localhost:5001';

// Текущая транзакция (для сравнения)
let currentTransaction = null;

// ==================== УТИЛИТЫ ====================

function formatCurrency(amount) {
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'KZT',
        minimumFractionDigits: 0
    }).format(amount);
}

function formatDate(date) {
    return new Date(date).toLocaleString('ru-RU');
}

// ==================== API ЗАПРОСЫ ====================

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
        
        // Обновляем статистику на странице
        document.getElementById('totalTransactions').textContent = 
            stats.total_transactions.toLocaleString();
        document.getElementById('fraudDetected').textContent = 
            stats.fraud_detected.toLocaleString();
        document.getElementById('accuracyRate').textContent = 
            (stats.accuracy * 100).toFixed(1) + '%';
    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
    }
}

// ==================== ЗАГРУЗКА ПРИМЕРОВ ТРАНЗАКЦИЙ ====================

async function loadSampleTransaction(type) {
    try {
        showLoading();
        
        // Получаем транзакцию из БД
        const response = await fetch(`${API_URL}/get_sample_transaction?type=${type}`);
        
        if (!response.ok) {
            throw new Error('Не удалось загрузить транзакцию');
        }
        
        const transaction = await response.json();
        
        // Сохраняем для сравнения
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
        
        // Показываем результат с реальной меткой
        showResultWithComparison(result, transaction);
        
    } catch (error) {
        showError(error.message);
    }
}

// Обработчики для кнопок примеров
window.loadCleanExample = () => loadSampleTransaction('clean');
window.loadFraudExample = () => loadSampleTransaction('fraud');
window.loadRandomExample = () => loadSampleTransaction('random');

// ==================== ЗАГРУЗКА СТАТИСТИКИ ====================

// ==================== ОБРАБОТКА ФОРМЫ ====================

document.getElementById('transactionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Собираем данные формы
    const formData = {
        customerId: document.getElementById('customerId').value,
        amount: document.getElementById('amount').value,
        recipientId: document.getElementById('recipientId').value
    };
    
    // Показываем состояние загрузки
    showLoading();
    
    try {
        // Отправляем запрос к API
        const result = await checkTransaction(formData);
        
        // Показываем результат
        showResult(result, formData);
        
    } catch (error) {
        showError(error.message);
    }
});

// ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====================

function showLoading() {
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.innerHTML = `
        <div class="loading-state">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Анализ транзакции...</p>
        </div>
    `;
    resultPanel.classList.remove('hidden');
}

function showResult(result, formData) {
    const resultPanel = document.getElementById('resultPanel');
    
    const riskPercentage = result.fraud_probability;
    const decision = result.decision.toLowerCase();
    
    // Проверяем, есть ли реальная метка
    const hasActualLabel = currentTransaction && currentTransaction.is_fraud !== undefined;
    
    resultPanel.innerHTML = `
        <div class="result-state">
            <!-- Сравнение с реальной меткой (если есть) -->
            ${hasActualLabel ? renderComparisonBanner() : ''}
            
            <!-- Индикатор риска -->
            <div class="risk-indicator">
                <div class="risk-gauge">
                    <svg class="gauge-svg" viewBox="0 0 200 120">
                        <path class="gauge-background" 
                              d="M 20 100 A 80 80 0 0 1 180 100"
                              fill="none" 
                              stroke-width="20"/>
                        <path class="gauge-fill" 
                              d="M 20 100 A 80 80 0 0 1 180 100"
                              fill="none" 
                              stroke="${getGaugeColor(riskPercentage)}"
                              stroke-width="20"
                              stroke-dasharray="${getGaugeDashArray(riskPercentage)}"
                              stroke-linecap="round"/>
                        <line class="gauge-needle"
                              x1="100" y1="100"
                              x2="100" y2="30"
                              stroke="#333"
                              stroke-width="3"
                              style="transform: rotate(${getGaugeRotation(riskPercentage)}deg)"/>
                    </svg>
                </div>
                <span class="risk-percentage" style="color: ${getGaugeColor(riskPercentage)}">
                    ${riskPercentage.toFixed(1)}%
                </span>
                <span class="risk-label">Вероятность мошенничества</span>
            </div>
            
            <!-- Решение -->
            <div class="decision-badge ${decision}">
                <i class="fas ${getDecisionIcon(decision)}"></i>
                <span>${result.decision_label}</span>
            </div>
            
            <!-- Объяснение -->
            <div class="explanation">
                <h4><i class="fas fa-info-circle"></i> Причина</h4>
                <p>${result.decision_reason}</p>
            </div>
            
            <!-- Детали транзакции -->
            <div class="transaction-details">
                <h4><i class="fas fa-file-invoice-dollar"></i> Детали транзакции</h4>
                <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
                    ${hasActualLabel ? `<p><strong>ID транзакции:</strong> ${currentTransaction.transaction_id}</p>` : ''}
                    <p><strong>Клиент:</strong> ${formData.customerId}</p>
                    <p><strong>Сумма:</strong> ${formatCurrency(formData.amount)}</p>
                    <p><strong>Получатель:</strong> ${formData.recipientId || 'Не указан'}</p>
                    <p><strong>Время:</strong> ${formatDate(hasActualLabel ? currentTransaction.datetime : new Date())}</p>
                    ${hasActualLabel ? `<p><strong>Реальная метка:</strong> <span style="font-weight: bold; color: ${currentTransaction.is_fraud ? '#ef4444' : '#10b981'}">${currentTransaction.actual_label}</span></p>` : ''}
                </div>
            </div>
            
            <!-- Факторы риска -->
            ${renderRiskFactors(result.risk_factors)}
            
            <!-- Feature Importance -->
            ${renderFeatureImportance(result.feature_importance)}
            
            <!-- Кнопки действий -->
            <div style="margin-top: 2rem;">
                <button class="btn btn-primary btn-large" onclick="resetForm()">
                    <i class="fas fa-plus"></i> Проверить другую транзакцию
                </button>
            </div>
        </div>
    `;
}

function showResultWithComparison(result, transaction) {
    currentTransaction = transaction;
    showResult(result, {
        customerId: transaction.customer_id,
        amount: transaction.amount,
        recipientId: transaction.recipient_id
    });
}

function renderComparisonBanner() {
    const isCorrect = checkPredictionCorrectness();
    
    if (isCorrect === null) return '';
    
    return `
        <div style="
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            background: ${isCorrect ? '#dcfce7' : '#fee2e2'};
            border-left: 4px solid ${isCorrect ? '#10b981' : '#ef4444'};
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <i class="fas fa-${isCorrect ? 'check-circle' : 'times-circle'}" style="color: ${isCorrect ? '#10b981' : '#ef4444'}; font-size: 1.5rem;"></i>
                <h4 style="margin: 0; color: ${isCorrect ? '#10b981' : '#ef4444'};">
                    ${isCorrect ? '✅ Модель правильно классифицировала!' : '❌ Модель ошиблась'}
                </h4>
            </div>
            <p style="margin: 0; color: #64748b;">
                <strong>Реальная метка:</strong> ${currentTransaction.actual_label}<br>
                <strong>Предсказание модели:</strong> ${getPredictionLabel()}
            </p>
        </div>
    `;
}

function checkPredictionCorrectness() {
    if (!currentTransaction) return null;
    
    const actualFraud = currentTransaction.is_fraud;
    
    // Получаем текущую вероятность из последнего результата
    const resultPanel = document.getElementById('resultPanel');
    const riskText = resultPanel.querySelector('.risk-percentage');
    if (!riskText) return null;
    
    const probability = parseFloat(riskText.textContent);
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

function renderRiskFactors(factors) {
    if (!factors || factors.length === 0) return '';
    
    return `
        <div class="factors">
            <h4><i class="fas fa-exclamation-triangle"></i> Факторы риска</h4>
            ${factors.map(factor => `
                <div class="factor-item">
                    <div class="factor-icon ${factor.impact}">
                        <i class="fas fa-${factor.impact === 'increase' ? 'arrow-up' : 'arrow-down'}"></i>
                    </div>
                    <div class="factor-content">
                        <div class="factor-name">${translateFeatureName(factor.name)}</div>
                        <div class="factor-impact">
                            ${factor.impact === 'increase' ? 'Повышает' : 'Понижает'} риск
                        </div>
                    </div>
                    <div class="factor-value">${factor.value.toFixed(2)}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderFeatureImportance(importance) {
    if (!importance || importance.length === 0) return '';
    
    const maxImportance = Math.max(...importance.map(f => f.importance));
    
    return `
        <div class="shap-section">
            <h4><i class="fas fa-chart-bar"></i> Важность признаков</h4>
            ${importance.map(feature => {
                const percentage = (feature.importance / maxImportance) * 100;
                return `
                    <div class="shap-bar">
                        <div class="shap-bar-header">
                            <span class="shap-bar-label">${translateFeatureName(feature.feature)}</span>
                            <span class="shap-bar-value">${feature.importance}</span>
                        </div>
                        <div class="shap-bar-container">
                            <div class="shap-bar-fill positive" style="width: ${percentage}%"></div>
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
        <div class="error-state">
            <i class="fas fa-exclamation-circle"></i>
            <h3>Ошибка</h3>
            <p>${message}</p>
            <p style="color: #64748b; margin-top: 1rem;">
                Убедитесь, что ML API сервис запущен на http://localhost:5001
            </p>
            <button class="btn btn-secondary" onclick="resetForm()" style="margin-top: 1rem;">
                Попробовать снова
            </button>
        </div>
    `;
    resultPanel.classList.remove('hidden');
}

function resetForm() {
    document.getElementById('transactionForm').reset();
    document.getElementById('resultPanel').innerHTML = `
        <div style="text-align: center; color: #64748b; min-height: 400px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <i class="fas fa-arrow-left" style="font-size: 3rem; margin-bottom: 1rem;"></i>
            <p>Заполните форму и нажмите "Проверить транзакцию"</p>
        </div>
    `;
    currentTransaction = null;
}

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

function getGaugeColor(percentage) {
    if (percentage < 30) return '#10b981';
    if (percentage < 50) return '#f59e0b';
    if (percentage < 70) return '#ea580c';
    return '#ef4444';
}

function getGaugeDashArray(percentage) {
    const circumference = Math.PI * 160;
    const offset = circumference - (percentage / 100) * circumference;
    return `${circumference} ${circumference}`;
}

function getGaugeRotation(percentage) {
    return -90 + (percentage / 100) * 180;
}

function getDecisionIcon(decision) {
    const icons = {
        'approve': 'fa-check-circle',
        'verify': 'fa-search',
        'review': 'fa-exclamation-triangle',
        'block': 'fa-ban'
    };
    return icons[decision] || 'fa-question-circle';
}

function translateFeatureName(name) {
    const translations = {
        'amount': 'Сумма транзакции',
        'amount_log': 'Логарифм суммы',
        'is_night': 'Ночное время',
        'is_weekend': 'Выходной день',
        'burstiness_score': 'Показатель активности',
        'interval_zscore': 'Отклонение интервала',
        'logins_last_7_days': 'Входы за 7 дней',
        'avg_session_interval_sec': 'Средний интервал сессий',
        'hour': 'Час',
        'day_of_week': 'День недели'
    };
    return translations[name] || name;
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================

document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    
    // Проверка доступности API
    fetch(`${API_URL}/health`)
        .then(response => response.json())
        .then(data => {
            console.log('✅ ML API доступен:', data);
        })
        .catch(error => {
            console.error('❌ ML API недоступен:', error);
            alert('⚠️ ML API сервис недоступен. Запустите: python ml_api_service.py');
        });
});