import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.teacher import TeacherNet
from models.student import StudentNet
from src.load_data import get_dataloaders
from src.distill_utils import distillation_loss
from src.evaluate import evaluate_model
import csv

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
lr = 0.001
T = 3.0
alpha = 0.5
student_save_path = "models/student_model.pth"
teacher_path = "models/teacher_model.pth"

def log_metrics(epoch, loss, acc, csv_path):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['epoch', 'loss', 'accuracy'])
        writer.writerow([epoch + 1, round(loss, 4), round(acc, 2)])

def train(student, teacher, train_loader, test_loader):
    student.to(device)
    teacher.to(device)
    teacher.eval()

    optimizer = optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        student.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            student_outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        acc = evaluate_model(student, test_loader, device)
        log_metrics(epoch, total_loss, acc, "results/student_metrics.csv")

    torch.save(student.state_dict(), student_save_path)
    print(f"Student model saved to {student_save_path}")

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    student = StudentNet()
    teacher = TeacherNet()
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    train(student, teacher, train_loader, test_loader)