import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.teacher import TeacherNet
from src.load_data import get_dataloaders
from src.evaluate import evaluate_model
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
lr = 0.001
save_path = "models/teacher_model.pth"

def log_metrics(epoch, loss, acc, csv_path):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['epoch', 'loss', 'accuracy'])
        writer.writerow([epoch + 1, round(loss, 4), round(acc, 2)])

def train(model, train_loader, test_loader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        acc = evaluate_model(model, test_loader, device)
        log_metrics(epoch, total_loss, acc, "results/metrics.csv")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Teacher model saved to {save_path}")

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    model = TeacherNet()
    train(model, train_loader, test_loader)