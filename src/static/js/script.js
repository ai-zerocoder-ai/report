// src/static/js/script.js
class ChatApp {
    constructor() {
        this.input = document.getElementById('question-input');
        this.sendBtn = document.getElementById('send-btn');
        this.chatMessages = document.getElementById('chat-messages');
        this.status = document.getElementById('status');
        this.init();
    }

    init() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        this.input.focus();
        this.checkStatus();
        setInterval(() => this.checkStatus(), 30000);
    }

    async checkStatus() {
        try {
            this.setStatus('loading', 'Проверка статуса...');
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.rag_initialized) {
                this.setStatus('success', `Готов (документов: ${data.document_count})`);
            } else {
                this.setStatus('error', 'Индекс не готов. Нажмите "Пересобрать" или проверьте JSON.');
            }
        } catch {
            this.setStatus('error', 'Ошибка подключения');
        }
    }

    async sendMessage() {
        const question = this.input.value.trim();
        if (!question) return;

        this.addMessage(question, 'user');
        this.input.value = '';
        this.sendBtn.disabled = true;
        this.input.disabled = true;

        // Показать лоадер
        this.addMessage('Думаю…', 'bot loading-message');

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const data = await response.json();
            this.removeLoadingMessage();

            if (data.status === 'success') {
                this.addMessage(data.answer, 'bot');
            } else {
                this.addMessage('Ошибка: ' + (data.error || data.message), 'bot error');
            }
        } catch {
            this.removeLoadingMessage();
            this.addMessage('Ошибка сети', 'bot error');
        } finally {
            this.sendBtn.disabled = false;
            this.input.disabled = false;
            this.input.focus();
            this.scrollToBottom();
        }
    }

    addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    removeLoadingMessage() {
        this.chatMessages
            .querySelectorAll('.loading-message-message')
            .forEach(msg => msg.remove());
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    setStatus(type, message) {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
}

document.addEventListener('DOMContentLoaded', () => new ChatApp());
