// Example data for Precision, Recall & F1 Score for algorithms
const algorithmsData = [
    { algorithm: "AES-128", precision: 0.87, recall: 0.85, f1: 0.86 },
    { algorithm: "ARC4", precision: 0.88, recall: 0.84, f1: 0.86 },
    { algorithm: "Blake", precision: 0.85, recall: 0.83, f1: 0.84 },
    { algorithm: "Blowfish", precision: 0.81, recall: 0.80, f1: 0.80 },
    { algorithm: "Cast-128", precision: 0.79, recall: 0.77, f1: 0.78 },
    { algorithm: "ChaCha20", precision: 0.87, recall: 0.85, f1: 0.86 },
    { algorithm: "Keccak", precision: 0.88, recall: 0.84, f1: 0.86 },
    { algorithm: "Salsa20", precision: 0.85, recall: 0.83, f1: 0.84 },
    { algorithm: "Triple-DES", precision: 0.81, recall: 0.80, f1: 0.80 }
];

// Function to create and populate the Precision/Recall/F1 Score table
function generatePrecisionRecallTable() {
    let tableHtml = "<thead><tr><th>Algorithm</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead><tbody>";

    // Iterate over the algorithms data and create table rows
    algorithmsData.forEach(item => {
        tableHtml += `<tr>
                        <td>${item.algorithm}</td>
                        <td>${item.precision}</td>
                        <td>${item.recall}</td>
                        <td>${item.f1}</td>
                      </tr>`;
    });

    tableHtml += "</tbody>";
    document.getElementById('precisionRecallTable').innerHTML = tableHtml;
}

// Example data for Accuracy, Macro Avg, and Weighted Avg with Precision, Recall, and F1 Score
const modelMetricsData = {
    accuracy: { precision: null, recall: null, f1: 0.72 },
    macroAvg: { precision: 0.84, recall: 0.82, f1: 0.83 },
    weightedAvg: { precision: 0.85, recall: 0.83, f1: 0.84 }
};

// Function to check if the value exists or return a hyphen
function checkValue(value) {
    return value !== null ? value : "-";
}

// Fill model metrics table with data
function generateModelMetricsTable() {
    let tableHtml = "<thead><tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead><tbody>";

    // Accuracy Row
    tableHtml += `<tr><td>Accuracy</td><td>${checkValue(modelMetricsData.accuracy.precision)}</td><td>${checkValue(modelMetricsData.accuracy.recall)}</td><td>${checkValue(modelMetricsData.accuracy.f1)}</td></tr>`;
    
    // Macro Average Row
    tableHtml += `<tr><td>Macro Average</td><td>${checkValue(modelMetricsData.macroAvg.precision)}</td><td>${checkValue(modelMetricsData.macroAvg.recall)}</td><td>${checkValue(modelMetricsData.macroAvg.f1)}</td></tr>`;
    
    // Weighted Average Row
    tableHtml += `<tr><td>Weighted Average</td><td>${checkValue(modelMetricsData.weightedAvg.precision)}</td><td>${checkValue(modelMetricsData.weightedAvg.recall)}</td><td>${checkValue(modelMetricsData.weightedAvg.f1)}</td></tr>`;
    
    tableHtml += "</tbody>";
    document.getElementById('metricsTable').innerHTML = tableHtml;
}

// Plot the bar graph for Precision, Recall, and F1 Score
function generateBarGraph() {
    const precision = algorithmsData.map(item => item.precision);
    const recall = algorithmsData.map(item => item.recall);
    const f1 = algorithmsData.map(item => item.f1);
    const algorithms = algorithmsData.map(item => item.algorithm);

    const data = [
        {
            x: algorithms,
            y: precision,
            name: 'Precision',
            type: 'bar'
        },
        {
            x: algorithms,
            y: recall,
            name: 'Recall',
            type: 'bar'
        },
        {
            x: algorithms,
            y: f1,
            name: 'F1 Score',
            type: 'bar'
        }
    ];

    const layout = {
        barmode: 'group',
        title: 'Performance of Algorithms',
        xaxis: { title: 'Algorithms' },
        yaxis: { title: 'Scores' }
    };

    Plotly.newPlot('barGraph', data, layout);
}

// Call functions to render content
generateModelMetricsTable();
generatePrecisionRecallTable();
generateBarGraph();

