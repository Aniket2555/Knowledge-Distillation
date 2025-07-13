import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"✅ Accuracy: {acc:.2f}%")
    return acc
