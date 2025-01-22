window.onload = function() {
    const ctx = document.getElementById('chart').getContext('2d');
    
    const data = {
        labels: ['Alg1', 'Alg2', 'Alg3'],
        datasets: [{
            label: 'Algorithm Confidence',
            data: [88, 85, 90],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    };

    const chart = new Chart(ctx, config);
};

	
