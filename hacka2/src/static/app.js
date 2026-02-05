/**
 * RSS Intelligence - Dashboard Logic
 */

class Dashboard {
    constructor() {
        this.feedEl = document.getElementById('feed');
        this.chatBox = document.getElementById('chat-box');
        this.userInput = document.getElementById('user-input');
        this.articles = [];
        this.activeCategory = 'all';
        this.searchQuery = '';

        this.initCharts();
        this.initWebSocket();
        this.bindEvents();
        this.loadHistory();
    }

    bindEvents() {
        // Search
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.searchQuery = e.target.value.toLowerCase();
            this.filterFeed();
        });

        // Categories
        document.querySelectorAll('.pill').forEach(pill => {
            pill.addEventListener('click', () => {
                document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                this.activeCategory = pill.dataset.category;
                this.filterFeed();
            });
        });

        // Chat
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Refresh
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadHistory();
            // Animation for visual feedback
            const icon = document.querySelector('#refresh-btn svg');
            icon.style.transform = 'rotate(360deg)';
            setTimeout(() => icon.style.transform = '', 300);
        });
    }

    initWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws`);

        this.ws.onmessage = (event) => {
            const article = JSON.parse(event.data);
            this.articles.unshift(article);
            this.addArticleToUI(article, true);
            this.updateStats();
            this.updateCharts(article);
        };

        this.ws.onclose = () => {
            console.log('WS Connection lost. Reconnecting...');
            setTimeout(() => this.initWebSocket(), 3000);
        };
    }

    initCharts() {
        // Sentiment Trend Chart
        const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
        this.sentimentChart = new Chart(sentimentCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sentiment',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: -1, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Category Distribution Chart
        const categoryCtx = document.getElementById('categoryChart').getContext('2d');
        this.categoryChart = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#6366f1', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 10 }
                    }
                },
                cutout: '70%'
            }
        });

        // Category Comparison Bar Chart
        const barCtx = document.getElementById('categoryBarChart').getContext('2d');
        this.categoryBarChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Avg Sentiment',
                    data: [],
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: '#6366f1',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: -1, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    updateCharts(article) {
        // Update Sentiment
        const labels = this.sentimentChart.data.labels;
        const data = this.sentimentChart.data.datasets[0].data;

        labels.push('');
        data.push(article.sentiment_score);

        if (labels.length > 20) {
            labels.shift();
            data.shift();
        }
        this.sentimentChart.update('none');

        // Update Category Distribution
        this.updateCategoryAnalytics();
    }

    updateCategoryAnalytics() {
        const counts = {};
        const sentiments = {};

        this.articles.forEach(a => {
            const cat = a.category ? a.category.toLowerCase() : 'unknown';
            counts[cat] = (counts[cat] || 0) + 1;
            sentiments[cat] = (sentiments[cat] || 0) + a.sentiment_score;
        });

        const labels = Object.keys(counts);
        const countData = Object.values(counts);
        const sentimentData = labels.map(l => sentiments[l] / counts[l]);

        // Update Doughnut Chart
        this.categoryChart.data.labels = labels.map(l => l.toUpperCase());
        this.categoryChart.data.datasets[0].data = countData;
        this.categoryChart.update();

        // Update Bar Chart
        this.categoryBarChart.data.labels = labels.map(l => l.toUpperCase());
        this.categoryBarChart.data.datasets[0].data = sentimentData;
        this.categoryBarChart.update();
    }

    async loadHistory() {
        try {
            const resp = await fetch('/api/articles');
            this.articles = await resp.json();

            // Initial render
            this.feedEl.innerHTML = '';
            this.articles.forEach(a => this.addArticleToUI(a, false));

            // Initial charts and insights
            this.updateCategoryChart();
            this.updateStats();
            this.updateInsights();

            // Populate sentiment trend with historical data
            const historicalSentiments = this.articles.slice(-20).reverse();
            this.sentimentChart.data.labels = historicalSentiments.map(() => '');
            this.sentimentChart.data.datasets[0].data = historicalSentiments.map(a => a.sentiment_score);
            this.sentimentChart.update();

        } catch (e) {
            console.error('Failed to load history', e);
        }
    }

    async updateInsights() {
        try {
            const resp = await fetch('/api/insights');
            const data = await resp.json();

            // Update Trending Topics
            const topicsEl = document.getElementById('trending-topics');
            topicsEl.innerHTML = '';
            data.trending_keywords.forEach(kw => {
                const tag = document.createElement('div');
                tag.className = 'topic-tag';
                tag.innerText = kw._id;
                tag.title = `${kw.count} occurrences`;
                topicsEl.appendChild(tag);
            });

            // Update Intelligence Matrix
            const matrixEl = document.getElementById('intel-matrix');
            matrixEl.innerHTML = '';
            data.category_stats.forEach(cat => {
                const score = cat.avg_sentiment;
                let scoreClass = 'score-neu';
                if (score > 0.1) scoreClass = 'score-pos';
                if (score < -0.1) scoreClass = 'score-neg';

                const item = document.createElement('div');
                item.className = 'matrix-item';
                item.innerHTML = `
                    <span class="matrix-label">${cat._id}</span>
                    <span class="matrix-score ${scoreClass}">${score.toFixed(2)}</span>
                `;
                matrixEl.appendChild(item);
            });
        } catch (e) {
            console.error('Failed to update insights', e);
        }
    }

    filterFeed() {
        this.feedEl.innerHTML = '';
        const filtered = this.articles.filter(a => {
            const matchesCat = this.activeCategory === 'all' || (a.category && a.category.toLowerCase() === this.activeCategory);
            const matchesSearch = a.title.toLowerCase().includes(this.searchQuery) ||
                (a.summary && a.summary.toLowerCase().includes(this.searchQuery));
            return matchesCat && matchesSearch;
        });
        filtered.forEach(a => this.addArticleToUI(a, false));
    }

    addArticleToUI(article, isNew) {
        // Skip if it doesn't match current filter
        const matchesCat = this.activeCategory === 'all' || article.category === this.activeCategory;
        const matchesSearch = article.title.toLowerCase().includes(this.searchQuery);
        if (!matchesCat || !matchesSearch) return;

        const card = document.createElement('div');
        card.className = 'article-card';

        const sentimentClass = `sentiment-${article.sentiment_label.toLowerCase()}`;

        card.innerHTML = `
            <div class="article-header">
                <span class="category-tag">${article.category}</span>
                <span class="sentiment-badge ${sentimentClass}">${article.sentiment_label}</span>
            </div>
            <h2 class="article-title">${article.title}</h2>
            <p class="article-summary">${article.summary}</p>
            <div class="article-footer">
                <span><a href="${article.link}" target="_blank" style="color: var(--primary-light); text-decoration: none;">Read Full Story →</a></span>
                <span>${new Date(article.published).toLocaleTimeString()}</span>
            </div>
        `;

        if (isNew) {
            this.feedEl.prepend(card);
        } else {
            this.feedEl.appendChild(card);
        }
    }

    async updateStats() {
        try {
            const resp = await fetch('/api/stats');
            const data = await resp.json();
            document.getElementById('stat-total').innerText = data.total_articles;

            // Calculate avg sentiment
            if (this.articles.length > 0) {
                const avg = this.articles.reduce((acc, a) => acc + a.sentiment_score, 0) / this.articles.length;
                let label = 'Neutral';
                if (avg > 0.2) label = 'Positive';
                if (avg < -0.2) label = 'Negative';
                document.getElementById('stat-sentiment').innerText = label;
            }
        } catch (e) { }
    }

    async sendMessage() {
        const text = this.userInput.value.trim();
        if (!text) return;

        this.userInput.value = '';
        this.appendMessage('user', text);

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await resp.json();
            this.appendMessage('bot', data.answer, data.sources);
        } catch (e) {
            this.appendMessage('bot', 'Sorry, I encountered an error. Please try again.');
        }
    }

    appendMessage(role, text, sources = []) {
        const msg = document.createElement('div');
        msg.className = `message ${role}`;

        if (role === 'bot') {
            // Render Markdown
            msg.innerHTML = marked.parse(text);

            // Add Sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'chat-sources';

                // Deduplicate sources by title
                const seen = new Set();
                sources.forEach(s => {
                    if (seen.has(s.title)) return;
                    seen.add(s.title);

                    const badge = document.createElement('a');
                    badge.href = s.link;
                    badge.target = '_blank';
                    badge.className = 'source-badge';
                    badge.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>${s.title.substring(0, 30)}...`;
                    sourcesDiv.appendChild(badge);
                });
                msg.appendChild(sourcesDiv);
            }
        } else {
            msg.innerText = text;
        }

        this.chatBox.appendChild(msg);
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
