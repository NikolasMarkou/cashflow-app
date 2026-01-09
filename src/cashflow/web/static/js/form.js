/**
 * Form handling and result display for cashflow forecasting web interface.
 */

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('forecast-form');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const submitBtn = document.getElementById('submit-btn');
    const horizonSlider = document.getElementById('forecast_horizon');
    const horizonValue = document.getElementById('horizon-value');
    const errorBanner = document.getElementById('error-message');

    // Update slider display value
    if (horizonSlider && horizonValue) {
        horizonSlider.addEventListener('input', function() {
            horizonValue.textContent = this.value;
        });
    }

    // Form submission handler
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            await runForecast();
        });
    }

    async function runForecast() {
        // Hide previous results and errors
        hideError();
        resultsSection.classList.add('hidden');

        // Show loading state
        loadingDiv.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').classList.add('hidden');
        submitBtn.querySelector('.btn-loading').classList.remove('hidden');

        // Build FormData
        const formData = new FormData(form);

        // Handle checkboxes for models_to_evaluate (FastAPI expects multiple values)
        const models = [];
        document.querySelectorAll('input[name="models_to_evaluate"]:checked')
            .forEach(cb => models.push(cb.value));

        // If no models selected, use defaults
        if (models.length === 0) {
            models.push('ets', 'sarima');
        }

        // Clear existing and re-add
        formData.delete('models_to_evaluate');
        models.forEach(m => formData.append('models_to_evaluate', m));

        try {
            const response = await fetch('/api/forecast', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP ${response.status}: Forecast failed`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            showError(error.message);
        } finally {
            // Reset loading state
            loadingDiv.classList.add('hidden');
            submitBtn.disabled = false;
            submitBtn.querySelector('.btn-text').classList.remove('hidden');
            submitBtn.querySelector('.btn-loading').classList.add('hidden');
        }
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Update metrics cards
        updateMetric('metric-wmape', `${data.wmape_winner.toFixed(2)}%`,
            data.meets_threshold ? 'success' : 'warning');
        updateMetric('metric-model', data.model_selected);
        updateMetric('metric-threshold', data.meets_threshold ? 'Yes' : 'No',
            data.meets_threshold ? 'success' : 'error');
        updateMetric('metric-confidence', data.confidence_level);

        // Render charts
        renderTimeSeriesChart('chart-timeseries', data.chart_data);
        renderModelComparisonChart('chart-model-comparison',
            data.model_candidates, data.wmape_threshold);
        renderComponentsChart('chart-components', data.chart_data.components);
        renderOutlierChart('chart-outliers', data.chart_data.outliers);

        // Update statistics table
        updateStatsTable(data);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function updateMetric(id, value, status) {
        const card = document.getElementById(id);
        if (!card) return;

        const valueSpan = card.querySelector('.metric-value');
        if (valueSpan) {
            valueSpan.textContent = value;
        }

        // Update status class
        card.classList.remove('success', 'warning', 'error');
        if (status) {
            card.classList.add(status);
        }
    }

    function updateStatsTable(data) {
        const tbody = document.getElementById('stats-body');
        if (!tbody) return;

        const formatNumber = (num) => {
            if (typeof num !== 'number') return num;
            return num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        };

        const stats = [
            ['Average NECF', formatNumber(data.decomposition_summary.avg_necf)],
            ['Average Deterministic Base', formatNumber(data.decomposition_summary.avg_deterministic_base)],
            ['Average Residual', formatNumber(data.decomposition_summary.avg_residual)],
            ['Transfers Removed', data.transfer_netting_summary.num_transfers_removed],
            ['Transfer Volume Removed', formatNumber(data.transfer_netting_summary.total_volume_removed)],
            ['Outliers Detected', data.outliers_detected.length],
            ['Forecast Period', `${data.forecast_start} to ${data.forecast_end}`],
            ['Horizon Months', data.horizon_months],
        ];

        tbody.innerHTML = stats.map(([label, value]) =>
            `<tr><td>${label}</td><td>${value}</td></tr>`
        ).join('');
    }

    function showError(message) {
        if (!errorBanner) return;

        errorBanner.innerHTML = `
            <div class="error-content">
                <strong>Error:</strong> ${escapeHtml(message)}
            </div>
            <button type="button" class="error-dismiss" onclick="this.parentElement.classList.add('hidden')">
                &times;
            </button>
        `;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        if (errorBanner) {
            errorBanner.classList.add('hidden');
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
