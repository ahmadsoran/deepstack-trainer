# DeepStack Custom Object Detection Training

DeepStack is a cross platform AI engine for performing Object Detection, Face Detection and Face Recognition on the edge and the cloud.

This repo provides functionality to train object detection models on your own objects, the model from this can be instantly deployed
to DeepStack on Desktop devices, Nvidia Jetson devices and the cloud.

Follow the instructions in this repo for setup, preparing your dataset, training your model and deploying for production use with DeepStack.

## 🛡️ Security & Code Quality

This project follows security best practices and maintains high code quality standards:

- **Security Auditing**: Regular dependency vulnerability scanning
- **Code Quality**: Automated linting, formatting, and type checking
- **Testing**: Comprehensive test coverage
- **Pre-commit Hooks**: Automated quality checks before commits

### Quick Security & Quality Setup

```bash
# Set up development environment with all quality tools
python setup_dev.py

# Run security audit
python security_audit.py

# Run code quality checks
python quality_check.py
```

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+ (with CUDA support recommended)
- Git

### Clone DeepStack Trainer
```bash
git clone https://github.com/johnolafenwa/deepstack-trainer
cd deepstack-trainer
```

### Install Requirements
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Set up development environment
python setup_dev.py
```

### Verify Installation
```bash
# Test basic functionality
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 📊 Training Your Model

### Basic Training Command
```bash
python train.py --dataset-path /path/to/your/dataset --epochs 100
```

### Advanced Training Options
```bash
python train.py \
    --dataset-path /path/to/dataset \
    --model yolov5m \
    --epochs 300 \
    --batch-size 16 \
    --img-size 640 \
    --hyp data/hyp.scratch.yaml
```

## 🔍 Detection and Testing

### Run Detection
```bash
python detect.py --weights yolov5m.pt --source data/images
```

### Test Model Performance
```bash
python test.py --weights yolov5m.pt --data data/coco.yaml
```

## 🛠️ Development

### Code Quality Tools
- **Black**: Code formatting
- **Flake8**: Style linting
- **MyPy**: Type checking
- **Bandit**: Security linting
- **Pytest**: Testing framework

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📁 Project Structure

```
deepstack-trainer/
├── data/                    # Configuration files
├── models/                  # Model definitions
├── utils/                   # Utility functions
├── train.py                 # Training script
├── detect.py                # Detection script
├── test.py                  # Testing script
├── requirements.txt         # Dependencies
├── security_audit.py       # Security checking
├── quality_check.py        # Code quality checking
└── setup_dev.py            # Development setup
```

## 🔒 Security Features

- **Dependency Scanning**: Automated vulnerability detection
- **Secure Model Loading**: Safe checkpoint loading with validation
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Robust error handling and logging
- **File Permissions**: Proper file permission management

## 📈 Performance Optimization

- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Data Loading Optimization**: Efficient data loading with multiple workers
- **Memory Management**: Proper GPU memory management
- **Model Optimization**: Optimized model architectures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks: `python quality_check.py`
5. Run security audit: `python security_audit.py`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the [Issues](https://github.com/johnolafenwa/deepstack-trainer/issues) page
- Review the documentation
- Run diagnostic scripts: `python security_audit.py` and `python quality_check.py`

