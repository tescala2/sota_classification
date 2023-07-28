import numpy as np
import torch
import torch.nn as nn

from sota_classification.utils.data import get_loaders


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    num_batches = len(dataloader)

    total_loss = 0
    for i, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches

    return avg_loss


def evaluate(dataloader, model, criterion, device):
    model.eval()
    num_batches = len(dataloader)

    preds = []
    labels = []
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            output = model(X)
            total_loss += criterion(output, y).item()

            preds.extend([pred.item() for pred in output])
            labels.extend([label.item() for label in y])

    avg_loss = total_loss / num_batches

    return avg_loss


def run(path, model, name, device, transform=None, epochs=10, lr=0.001, bs=16):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf  # evaluate(val_loader, model, criterion, device, debug=True, mode='Val')
    # print(f"Initial Validation Loss: {best_val_loss:.5f}")

    for epoch in range(epochs):
        train_loader, val_loader, test_loader = get_loaders(path=path,
                                                           bs=bs,
                                                           shuffle=True,
                                                           transform=transform)

        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss = evaluate(val_loader, model, criterion, device)

        info = f"Epoch {epoch+1}: Training Loss: {train_loss:.5f}\tValidation Loss: {val_loss:.5f}"

        if best_val_loss > val_loss:
            torch.save(model.state_dict(), name + '.pth')
            best_val_loss = val_loss

            info += "\tCheckpoint!"

            test_loss = evaluate(test_loader, model, criterion, device)

        print(info)