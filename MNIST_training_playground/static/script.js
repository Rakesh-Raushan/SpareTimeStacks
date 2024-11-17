// Model training state
let trainingInProgress = false;
let model1Data = { loss: [], accuracy: [] };
let model2Data = { loss: [], accuracy: [] };
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize charts
function initCharts() {
    Plotly.newPlot('lossChart', [
        {
            y: [],
            type: 'line',
            name: 'Model 1 Loss'
        },
        {
            y: [],
            type: 'line',
            name: 'Model 2 Loss'
        }
    ], {
        title: 'Training Loss',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Loss' }
    });

    Plotly.newPlot('accuracyChart', [
        {
            y: [],
            type: 'line',
            name: 'Model 1 Accuracy'
        },
        {
            y: [],
            type: 'line',
            name: 'Model 2 Accuracy'
        }
    ], {
        title: 'Model Accuracy',
        xaxis: { title: 'Iteration' },
        yaxis: { 
            title: 'Accuracy (%)',
            range: [90, 100]
        }
    });
}

// Get model configurations
function getModelConfig(modelNum) {
    return {
        channels: [
            parseInt(document.getElementById(`m${modelNum}-ch1`).value),
            parseInt(document.getElementById(`m${modelNum}-ch2`).value),
            parseInt(document.getElementById(`m${modelNum}-ch3`).value),
            parseInt(document.getElementById(`m${modelNum}-ch4`).value)
        ],
        optimizer: document.getElementById(`m${modelNum}-optimizer`).value,
        batch_size: parseInt(document.getElementById(`m${modelNum}-batch-size`).value),
        epochs: parseInt(document.getElementById(`m${modelNum}-epochs`).value)
    };
}

// Start training
async function startTraining() {
    const trainButton = document.getElementById('train-button');
    trainButton.disabled = true;
    trainButton.textContent = 'Training in Progress...';

    // Clear previous data
    model1Data = { loss: [], accuracy: [] };
    model2Data = { loss: [], accuracy: [] };
    
    // Get configurations
    const config = {
        model1: getModelConfig(1),
        model2: getModelConfig(2)
    };

    try {
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        if (result.status === 'success') {
            trainingInProgress = true;
            startMonitoring();
        } else {
            alert('Error starting training: ' + result.message);
            resetTrainButton();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error starting training');
        resetTrainButton();
    }
}

function resetTrainButton() {
    const trainButton = document.getElementById('train-button');
    trainButton.disabled = false;
    trainButton.textContent = 'Train Models';
}

// Update charts with new data
function updateCharts() {
    fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            model1Data = data.model1;
            model2Data = data.model2;

            Plotly.update('lossChart', {
                y: [model1Data.loss, model2Data.loss]
            });

            Plotly.update('accuracyChart', {
                y: [model1Data.accuracy, model2Data.accuracy]
            });
        });
}

// Monitor training status
function checkTrainingStatus() {
    fetch('/training_status')
        .then(response => response.json())
        .then(data => {
            if (!data.in_progress && trainingInProgress) {
                trainingInProgress = false;
                resetTrainButton();
                displayTestExamples();
            }
        });
}

// Display test examples
function displayTestExamples() {
    fetch('/test_examples')
        .then(response => response.json())
        .then(data => {
            if (!data.model1 || !data.model2) return;
            
            const container = document.getElementById('testExamples');
            container.innerHTML = '<h2>Test Examples</h2>';
            
            const grid = document.createElement('div');
            grid.className = 'test-examples-grid';
            
            // Combine examples from both models
            data.model1.forEach((example, index) => {
                const div = document.createElement('div');
                div.className = 'test-example';
                
                const img = document.createElement('img');
                img.src = example.image_path;
                img.width = 28 * 5;
                img.height = 28 * 5;
                img.className = 'digit-image';
                
                div.appendChild(img);
                
                const correct1 = example.label === example.prediction;
                const correct2 = data.model2[index].label === data.model2[index].prediction;
                
                div.innerHTML += `
                    <p class="prediction-info">
                        <span class="label">True: ${example.label}</span><br>
                        <span class="prediction ${correct1 ? 'correct' : 'incorrect'}">
                            M1: ${example.prediction}
                        </span>
                        <span class="prediction ${correct2 ? 'correct' : 'incorrect'}">
                            M2: ${data.model2[index].prediction}
                        </span>
                    </p>
                `;
                
                grid.appendChild(div);
            });
            
            container.appendChild(grid);
        })
        .catch(error => {
            console.error('Error displaying test examples:', error);
        });
}

function startMonitoring() {
    // Update charts every 2 seconds
    const chartInterval = setInterval(() => {
        if (!trainingInProgress) {
            clearInterval(chartInterval);
            return;
        }
        updateCharts();
    }, 2000);

    // Check training status every 5 seconds
    const statusInterval = setInterval(() => {
        if (!trainingInProgress) {
            clearInterval(statusInterval);
            return;
        }
        checkTrainingStatus();
    }, 5000);
}

// Drawing board functionality
function initDrawingBoard() {
    const canvas = document.getElementById('drawingBoard');
    const ctx = canvas.getContext('2d');
    
    // Set up drawing style
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    
    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch event listeners for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Clear button
    document.getElementById('clearButton').addEventListener('click', clearCanvas);
    
    // Predict button
    document.getElementById('predictButton').addEventListener('click', getPrediction);
}

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    const canvas = document.getElementById('drawingBoard');
    const ctx = canvas.getContext('2d');
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function handleTouch(e) {
    e.preventDefault();
    const canvas = document.getElementById('drawingBoard');
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
        clientX: touch.clientX - rect.left,
        clientY: touch.clientY - rect.top
    });
    
    canvas.dispatchEvent(mouseEvent);
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    const canvas = document.getElementById('drawingBoard');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset predictions
    document.getElementById('prediction1').textContent = '-';
    document.getElementById('prediction2').textContent = '-';
    document.getElementById('confidence1').textContent = 'Confidence: -';
    document.getElementById('confidence2').textContent = 'Confidence: -';
}

function getPrediction() {
    const canvas = document.getElementById('drawingBoard');
    const ctx = canvas.getContext('2d');
    
    // Create a temporary canvas for preprocessing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw and resize to 28x28
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data and convert to grayscale
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = new Float32Array(28 * 28);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
        data[i/4] = (255 - imageData.data[i]) / 255.0;  // Convert to 0-1 range and invert
    }
    
    // Send to server for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: Array.from(data)
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(result => {
        if (result.error) {
            // Handle server-side error
            document.getElementById('prediction1').textContent = 'Error';
            document.getElementById('prediction2').textContent = 'Error';
            document.getElementById('confidence1').textContent = result.error;
            document.getElementById('confidence2').textContent = result.error;
        } else {
            // Update predictions
            document.getElementById('prediction1').textContent = result.model1.prediction;
            document.getElementById('prediction2').textContent = result.model2.prediction;
            document.getElementById('confidence1').textContent = 
                `Confidence: ${(result.model1.confidence * 100).toFixed(1)}%`;
            document.getElementById('confidence2').textContent = 
                `Confidence: ${(result.model2.confidence * 100).toFixed(1)}%`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        // Update UI to show error
        document.getElementById('prediction1').textContent = 'Error';
        document.getElementById('prediction2').textContent = 'Error';
        document.getElementById('confidence1').textContent = 'Failed to get prediction';
        document.getElementById('confidence2').textContent = 'Failed to get prediction';
    });
}

// Add this function after other function definitions and before the DOMContentLoaded event listener

function initLogStreams() {
    ['model1', 'model2'].forEach(modelId => {
        const logWindow = document.getElementById(`${modelId}-log`);
        const eventSource = new EventSource(`/model_log/${modelId}`);
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'heartbeat') return;
            
            const logLine = document.createElement('div');
            logLine.className = `log-line log-${data.type}`;
            logLine.textContent = data.message;
            
            logWindow.appendChild(logLine);
            logWindow.scrollTop = logWindow.scrollHeight;
            
            // Keep only last 100 lines
            while (logWindow.children.length > 100) {
                logWindow.removeChild(logWindow.firstChild);
            }
        };

        eventSource.onerror = function(error) {
            console.error('EventSource failed:', error);
            eventSource.close();
        };
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initLogStreams();
    initDrawingBoard();
    document.getElementById('train-button').addEventListener('click', startTraining);
}); 