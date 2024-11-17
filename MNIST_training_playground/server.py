from flask import Flask, render_template, jsonify, request, Response
import json
import os
import torch.multiprocessing as mp
from train import train_model, CNN
import threading
import queue
import torch
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)

# Global variables to store training states
model1_metrics = {'loss': [], 'accuracy': []}
model2_metrics = {'loss': [], 'accuracy': []}
training_in_progress = False

# Queues for log messages
model1_log_queue = queue.Queue()
model2_log_queue = queue.Queue()

# Add counter for completed models
active_training_threads = 0

# Add these global variables to store the models
model1 = None
model2 = None

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_id):
    try:
        model_path = f'static/models/{model_id}.pth'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        # Load the saved config from the training phase
        config_path = f'static/models/{model_id}_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            channels = config['channels']
        else:
            # Fallback to default configuration
            channels = [32, 64, 64, 64]
            
        # Create model with the same architecture
        model = CNN(channels).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

@app.route('/model_log/<model_id>')
def stream_logs(model_id):
    def generate():
        log_queue = model1_log_queue if model_id == 'model1' else model2_log_queue
        while True:
            try:
                log_entry = log_queue.get(timeout=1)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

def log_writer(message, model_id):
    queue = model1_log_queue if model_id == 'model1' else model2_log_queue
    queue.put({
        'message': message,
        'type': 'progress' if 'Epoch' in message else 'metric'
    })

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_in_progress, active_training_threads
    if training_in_progress:
        return jsonify({'status': 'error', 'message': 'Training already in progress'})
    
    training_in_progress = True
    active_training_threads = 2  # We're starting 2 models
    config = request.json
    
    # Clear previous metrics and logs
    model1_metrics.clear()
    model2_metrics.clear()
    model1_metrics.update({'loss': [], 'accuracy': []})
    model2_metrics.update({'loss': [], 'accuracy': []})
    
    while not model1_log_queue.empty():
        model1_log_queue.get()
    while not model2_log_queue.empty():
        model2_log_queue.get()
    
    def training_complete():
        global active_training_threads, training_in_progress
        active_training_threads -= 1
        if active_training_threads == 0:
            training_in_progress = False
    
    # Start training in separate threads
    thread1 = threading.Thread(
        target=lambda: [train_model(config['model1'], model1_metrics, 'model1', log_writer), training_complete()]
    )
    thread2 = threading.Thread(
        target=lambda: [train_model(config['model2'], model2_metrics, 'model2', log_writer), training_complete()]
    )
    
    thread1.start()
    thread2.start()
    
    return jsonify({'status': 'success', 'message': 'Training started'})

@app.route('/metrics')
def metrics():
    return jsonify({
        'model1': model1_metrics,
        'model2': model2_metrics
    })

@app.route('/training_status')
def training_status():
    return jsonify({
        'in_progress': training_in_progress
    })

@app.route('/test_examples_model1')
def test_examples_model1():
    try:
        with open('static/test_examples_model1.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])

@app.route('/test_examples_model2')
def test_examples_model2():
    try:
        with open('static/test_examples_model2.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])

@app.route('/test_examples')
def test_examples():
    try:
        examples1 = json.load(open('static/test_examples_model1.json', 'r'))
        examples2 = json.load(open('static/test_examples_model2.json', 'r'))
        return jsonify({
            'model1': examples1,
            'model2': examples2
        })
    except FileNotFoundError:
        return jsonify({
            'model1': [],
            'model2': []
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load models if not already loaded
        global model1, model2
        if model1 is None:
            model1 = load_model('model1')
        if model2 is None:
            model2 = load_model('model2')

        if model1 is None or model2 is None:
            return jsonify({
                'error': 'Models not trained yet or failed to load. Please train the models first.'
            }), 400

        # Get image data
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # Normalize the input similar to training data
        image = torch.tensor(data, dtype=torch.float32).reshape(1, 1, 28, 28)
        image = (image - 0.1307) / 0.3081  # Apply MNIST normalization
        image = image.to(device)

        # Get predictions from both models
        with torch.no_grad():
            try:
                # Model 1 prediction
                output1 = model1(image)
                probs1 = F.softmax(output1, dim=1)
                pred1 = torch.argmax(output1, dim=1).item()
                conf1 = probs1[0][pred1].item()
                
                # Model 2 prediction
                output2 = model2(image)
                probs2 = F.softmax(output2, dim=1)
                pred2 = torch.argmax(output2, dim=1).item()
                conf2 = probs2[0][pred2].item()

                return jsonify({
                    'model1': {
                        'prediction': int(pred1),
                        'confidence': float(conf1)
                    },
                    'model2': {
                        'prediction': int(pred2),
                        'confidence': float(conf2)
                    }
                })
            except Exception as e:
                print(f"Prediction computation error: {e}")
                return jsonify({'error': 'Error computing predictions'}), 500
                
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 