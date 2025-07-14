# Knowledge Distillation

Knowledge Distillation is a technique in deep learning where a smaller, student model learns to mimic a larger, pre-trained teacher model. This repository provides a complete pipeline for training, evaluating, and analyzing knowledge distillation experiments using PyTorch.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Analysis](#results--analysis)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project demonstrates how to:
- Train a large teacher neural network.
- Train a smaller student network using knowledge distillation.
- Evaluate and compare the performance of both models.
- Log and analyze results.

## Features
- Modular code for easy experimentation.
- Customizable distillation loss and hyperparameters.
- Training and evaluation scripts for both teacher and student models.
- Result logging and analysis via Jupyter notebooks.

## Project Structure
```
Knowledge-Distillation/
├── data/                # Dataset and data loaders
├── models/              # Model definitions (teacher & student)
│   ├── teacher.py
│   └── student.py
├── notebooks/           # Jupyter notebooks for analysis
│   └── analysis.ipynb
├── src/                 # Training, evaluation, and utility scripts
│   ├── distill_utils.py
│   ├── evaluate.py
│   ├── load_data.py
│   ├── train_student,py
│   └── train_teacher.py
├── results.py/          # (Possibly results scripts or outputs)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── venv/                # Python virtual environment (optional)
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Knowledge-Distillation.git
   cd Knowledge-Distillation
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Train the Teacher Model
```bash
python src/train_teacher.py
```
This will train the teacher model and save its weights to `models/teacher_model.pth`.

### 2. Train the Student Model with Distillation
```bash
python src/train_student,py
```
This will train the student model using knowledge distillation and save its weights to `models/student_model.pth`.

### 3. Evaluate Models
You can use the provided `src/evaluate.py` script or the Jupyter notebook in `notebooks/` to evaluate and compare model performance.

### 4. Analyze Results
Results and metrics are logged in `results/metrics.csv`. Use the Jupyter notebook for further analysis and visualization.

## Customization
- **Change Hyperparameters:** Edit the values in `src/train_teacher.py` and `src/train_student,py` (e.g., epochs, learning rate, temperature, alpha).
- **Modify Models:** Update `models/teacher.py` and `models/student.py` to experiment with different architectures.
- **Dataset:** Place your dataset in the `data/` directory and update `src/load_data.py` as needed.

## Results & Analysis
- Training and evaluation logs are saved in `results/metrics.csv`.
- Use `notebooks/analysis.ipynb` to visualize and analyze the results.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit them.
4. Push to your fork and submit a pull request.

Please open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
