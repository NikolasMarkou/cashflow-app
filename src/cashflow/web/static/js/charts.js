/**
 * Plotly.js chart rendering functions for cashflow forecasting.
 * Color palette matches existing visualization scripts.
 */

// Color palette matching existing visualization scripts
const COLORS = {
    actual: '#2E86AB',           // Steel blue - historical data
    forecast: '#28A745',         // Green - forecast projections
    ci_fill: 'rgba(40, 167, 69, 0.2)',  // Green transparent - CI bands
    ci_edge: '#28A745',
    outlier: '#DC3545',          // Red - detected outliers
    deterministic: '#17A2B8',    // Cyan - deterministic base
    residual: '#FFC107',         // Amber - residual component
    residual_negative: '#E57373', // Light red - negative residuals
    delta: '#6F42C1',            // Purple - future delta
    threshold: '#DC3545',        // Red - WMAPE threshold line
    winner: '#28A745',           // Green - winning model
    loser: '#6C757D',            // Gray - losing models
};

// Common layout settings
const LAYOUT_DEFAULTS = {
    paper_bgcolor: 'white',
    plot_bgcolor: '#FAFAFA',
    font: { family: 'system-ui, -apple-system, sans-serif' },
    margin: { t: 80, r: 30, b: 60, l: 70 },
};

/**
 * Chart 1: Historical + Forecast Time Series with Confidence Intervals
 */
function renderTimeSeriesChart(containerId, data) {
    const traces = [];

    // Historical data
    if (data.historical && data.historical.months.length > 0) {
        traces.push({
            x: data.historical.months,
            y: data.historical.necf,
            mode: 'lines+markers',
            name: 'Historical (Actual)',
            line: { color: COLORS.actual, width: 2.5 },
            marker: { size: 6, color: COLORS.actual },
        });
    }

    // Forecast confidence interval (upper bound - invisible line for fill)
    if (data.forecast && data.forecast.months.length > 0) {
        traces.push({
            x: data.forecast.months,
            y: data.forecast.upper_ci,
            mode: 'lines',
            name: 'Upper CI',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
        });

        // Forecast confidence interval (lower bound + fill)
        traces.push({
            x: data.forecast.months,
            y: data.forecast.lower_ci,
            mode: 'lines',
            name: '95% Confidence Interval',
            fill: 'tonexty',
            fillcolor: COLORS.ci_fill,
            line: { width: 0 },
        });

        // Forecast line
        traces.push({
            x: data.forecast.months,
            y: data.forecast.totals,
            mode: 'lines+markers',
            name: 'Forecast',
            line: { color: COLORS.forecast, width: 2.5 },
            marker: { size: 7, color: COLORS.forecast, symbol: 'square' },
        });
    }

    // Outlier markers on historical data
    if (data.historical && data.historical.outlier_months && data.historical.outlier_months.length > 0) {
        traces.push({
            x: data.historical.outlier_months,
            y: data.historical.outlier_values,
            mode: 'markers',
            name: 'Detected Outliers',
            marker: {
                color: COLORS.outlier,
                size: 12,
                symbol: 'circle-open',
                line: { width: 3, color: COLORS.outlier },
            },
        });
    }

    // Determine forecast start for vertical line
    let forecastStartX = null;
    if (data.forecast && data.forecast.months.length > 0) {
        forecastStartX = data.forecast.months[0];
    }

    const layout = {
        ...LAYOUT_DEFAULTS,
        title: {
            text: `Cash Flow Forecast - ${data.forecast ? data.forecast.months.length : 0} Month Horizon`,
            font: { size: 16 },
        },
        xaxis: {
            title: 'Month',
            tickangle: -45,
            gridcolor: '#E0E0E0',
        },
        yaxis: {
            title: 'Net External Cash Flow',
            zeroline: true,
            zerolinecolor: '#CCCCCC',
            gridcolor: '#E0E0E0',
        },
        legend: {
            x: 0,
            y: 1.02,
            orientation: 'h',
            bgcolor: 'rgba(255,255,255,0.8)',
        },
        shapes: forecastStartX ? [{
            type: 'line',
            x0: forecastStartX,
            x1: forecastStartX,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#6C757D', width: 1.5, dash: 'dash' },
        }] : [],
        annotations: forecastStartX ? [{
            x: forecastStartX,
            y: 1.02,
            yref: 'paper',
            text: 'Forecast Start',
            showarrow: false,
            font: { size: 10, color: '#6C757D' },
        }] : [],
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Chart 2: Model Comparison Bar Chart
 */
function renderModelComparisonChart(containerId, candidates, threshold) {
    if (!candidates || candidates.length === 0) {
        document.getElementById(containerId).innerHTML = '<p class="no-data">No model candidates available</p>';
        return;
    }

    const colors = candidates.map(c => c.is_winner ? COLORS.winner : COLORS.loser);

    const traces = [{
        x: candidates.map(c => c.model_name),
        y: candidates.map(c => c.wmape),
        type: 'bar',
        marker: { color: colors },
        text: candidates.map(c =>
            `${c.wmape.toFixed(2)}%${c.is_winner ? ' (Winner)' : ''}`
        ),
        textposition: 'outside',
        textfont: { size: 11 },
    }];

    const maxWmape = Math.max(...candidates.map(c => c.wmape));

    const layout = {
        ...LAYOUT_DEFAULTS,
        title: {
            text: 'Model Selection: WMAPE Comparison',
            font: { size: 14 },
        },
        xaxis: { title: 'Model' },
        yaxis: {
            title: 'WMAPE (%)',
            range: [0, Math.max(maxWmape * 1.3, threshold * 1.2)],
            gridcolor: '#E0E0E0',
        },
        shapes: [{
            type: 'line',
            x0: -0.5,
            x1: candidates.length - 0.5,
            y0: threshold,
            y1: threshold,
            line: { color: COLORS.threshold, width: 2, dash: 'dash' },
        }],
        annotations: [{
            x: candidates.length - 0.5,
            y: threshold,
            text: `Threshold (${threshold}%)`,
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { color: COLORS.threshold, size: 11 },
        }],
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Chart 3: Component Breakdown (Stacked Bar)
 */
function renderComponentsChart(containerId, components) {
    if (!components || !components.months || components.months.length === 0) {
        document.getElementById(containerId).innerHTML = '<p class="no-data">No component data available</p>';
        return;
    }

    // Separate positive and negative residuals
    const residualPos = components.residual.map(r => Math.max(0, r));
    const residualNeg = components.residual.map(r => Math.min(0, r));

    const traces = [
        {
            x: components.months,
            y: components.deterministic_base,
            name: 'Deterministic Base',
            type: 'bar',
            marker: { color: COLORS.deterministic },
        },
        {
            x: components.months,
            y: residualPos,
            name: 'Residual (+)',
            type: 'bar',
            marker: { color: COLORS.residual },
        },
        {
            x: components.months,
            y: residualNeg,
            name: 'Residual (-)',
            type: 'bar',
            marker: { color: COLORS.residual_negative },
        },
        {
            x: components.months,
            y: components.totals,
            name: 'Forecast Total',
            mode: 'lines+markers',
            type: 'scatter',
            marker: { symbol: 'diamond', size: 8, color: '#212529' },
            line: { color: '#212529', width: 2.5 },
        },
    ];

    // Add delta if present and non-zero
    if (components.delta && components.delta.some(d => d !== 0)) {
        traces.splice(3, 0, {
            x: components.months,
            y: components.delta,
            name: 'Known Future Delta',
            type: 'bar',
            marker: { color: COLORS.delta },
        });
    }

    const layout = {
        ...LAYOUT_DEFAULTS,
        title: {
            text: 'Forecast = Deterministic + Residual + Delta',
            font: { size: 14 },
        },
        barmode: 'relative',
        xaxis: {
            title: 'Forecast Month',
            tickangle: -45,
        },
        yaxis: {
            title: 'Cash Flow',
            gridcolor: '#E0E0E0',
        },
        legend: {
            x: 0,
            y: 1.02,
            orientation: 'h',
            bgcolor: 'rgba(255,255,255,0.8)',
        },
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Chart 4: Outlier Analysis
 */
function renderOutlierChart(containerId, outliers) {
    if (!outliers || outliers.length === 0) {
        document.getElementById(containerId).innerHTML =
            '<div class="no-data"><p>No outliers detected</p><small>The residual series has no anomalous values above the threshold.</small></div>';
        return;
    }

    const traces = [
        {
            x: outliers.map(o => o.month_key),
            y: outliers.map(o => o.original_value),
            name: 'Original Value',
            type: 'bar',
            marker: { color: COLORS.outlier },
        },
        {
            x: outliers.map(o => o.month_key),
            y: outliers.map(o => o.treated_value),
            name: 'Treated Value',
            type: 'bar',
            marker: { color: COLORS.deterministic },
        },
    ];

    const layout = {
        ...LAYOUT_DEFAULTS,
        title: {
            text: 'Outlier Detection & Treatment',
            font: { size: 14 },
        },
        barmode: 'group',
        xaxis: {
            title: 'Month',
            tickangle: -45,
        },
        yaxis: {
            title: 'Residual Value',
            gridcolor: '#E0E0E0',
        },
        legend: {
            x: 0,
            y: 1.02,
            orientation: 'h',
        },
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

// Export for use in form.js
window.renderTimeSeriesChart = renderTimeSeriesChart;
window.renderModelComparisonChart = renderModelComparisonChart;
window.renderComponentsChart = renderComponentsChart;
window.renderOutlierChart = renderOutlierChart;
