# Training Monitoring System

A comprehensive machine learning training monitoring system that provides real-time tracking, visualization, and management of ML model training processes.

## 🚀 Features

- **Real-time Training Monitoring**: Track training progress, metrics, and performance in real-time
- **Interactive Dashboards**: Visualize training metrics, loss curves, and model performance
- **Experiment Tracking**: Compare different training runs and experiments
- **Alert System**: Get notified when training completes or encounters issues
- **Resource Monitoring**: Track GPU/CPU usage, memory consumption, and system resources
- **Model Management**: Save, version, and manage trained models
- **Team Collaboration**: Share training results and collaborate with team members

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- CUDA-compatible GPU (optional, for GPU training)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://rachanagupta@bitbucket.org/simformteam/ml-poc.git
   cd ml-poc
   git checkout training-monitoring-system
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

## 🏃‍♂️ Quick Start

1. **Start the monitoring server**:
   ```bash
   python app.py
   ```

2. **Access the dashboard**:
   Open your browser and navigate to `http://localhost:5000`

3. **Start training with monitoring**:
   ```python
   from training_monitor import TrainingMonitor
   
   monitor = TrainingMonitor(experiment_name="my_experiment")
   
   # Your training loop
   for epoch in range(epochs):
       # Training code here...
       
       # Log metrics
       monitor.log_metrics({
           'loss': train_loss,
           'accuracy': train_acc,
           'val_loss': val_loss,
           'val_accuracy': val_acc
       }, step=epoch)
   ```

## 📁 Project Structure

```
training_monitor_system/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── static/              # Static assets (CSS, JS, images)
├── templates/           # HTML templates
├── src/                 # Source code
│   ├── monitor/         # Training monitoring modules
│   ├── dashboard/       # Dashboard components
│   ├── utils/          # Utility functions
│   └── models/         # Data models
├── data/               # Training data and logs
├── experiments/        # Experiment results
└── tests/             # Unit tests
```

## 🔧 Configuration

The system can be configured using environment variables in the `.env` file:

```env
# Server Configuration
HOST=localhost
PORT=5000
DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///training_monitor.db

# Monitoring Configuration
LOG_LEVEL=INFO
METRICS_RETENTION_DAYS=30

# Notification Settings
ENABLE_NOTIFICATIONS=True
SLACK_WEBHOOK_URL=your_slack_webhook_url
EMAIL_NOTIFICATIONS=True
```

## 📊 Usage Examples

### Basic Training Monitoring

```python
from training_monitor import TrainingMonitor
import torch

# Initialize monitor
monitor = TrainingMonitor(
    experiment_name="resnet50_training",
    project_name="image_classification"
)

# Training loop
model = ResNet50()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    train_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Log batch metrics
        if batch_idx % 10 == 0:
            monitor.log_metrics({
                'batch_loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch * len(train_loader) + batch_idx)
    
    # Log epoch metrics
    monitor.log_metrics({
        'epoch_loss': train_loss / len(train_loader),
        'epoch': epoch
    }, step=epoch)
    
    # Save model checkpoint
    if epoch % 10 == 0:
        monitor.save_checkpoint(model.state_dict(), epoch)
```

### Advanced Monitoring with Visualization

```python
# Log custom visualizations
monitor.log_image("confusion_matrix", confusion_matrix_plot)
monitor.log_histogram("weights", model.layer1.weight.data)
monitor.log_graph(model, input_sample)

# Add tags and notes
monitor.add_tag("architecture", "resnet50")
monitor.add_note("Started training with increased learning rate")

# Compare experiments
monitor.compare_experiments(["exp1", "exp2", "exp3"])
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard home page |
| `/api/experiments` | GET | List all experiments |
| `/api/experiments/<id>` | GET | Get specific experiment |
| `/api/metrics/<experiment_id>` | GET | Get experiment metrics |
| `/api/logs/<experiment_id>` | GET | Get experiment logs |
| `/api/models/<experiment_id>` | GET | Get saved models |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_monitor.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on [Bitbucket Issues](https://bitbucket.org/simformteam/ml-poc/issues)
- **Documentation**: Check the [Wiki](https://bitbucket.org/simformteam/ml-poc/wiki) for detailed documentation
- **Team Contact**: Reach out to the development team for support

## 🎯 Roadmap

- [ ] Integration with popular ML frameworks (TensorFlow, PyTorch, Scikit-learn)
- [ ] Advanced hyperparameter optimization tracking
- [ ] Multi-user authentication and permissions
- [ ] Cloud deployment support (AWS, GCP, Azure)
- [ ] Mobile app for monitoring on-the-go
- [ ] Integration with MLOps platforms

## ⚡ Performance Tips

- Use GPU monitoring for CUDA-enabled training
- Set appropriate log levels to avoid performance overhead
- Configure metrics retention to manage storage
- Use batch logging for high-frequency metrics

---

**Built with ❤️ by the Simform ML Team**