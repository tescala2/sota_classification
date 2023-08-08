import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as ps

from torch import save, load
from .data import get_dataloader


def train(dataloader, model, criterion, optimizer, device, progress=False, parallel=False):
    if parallel:
        assert torch.cuda.device_count() > 1, "Attempted to use multiple GPUs, but only 1 GPU is available"
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.train()
    num_batches = len(dataloader)

    total_loss = 0
    total_correct = 0
    total_images = 0
    start = time.time()
    for i, (X, y) in enumerate(dataloader):

        X, y = X.float().to(device), y.to(device)

        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        scores, classes = output.max(dim=1)
        total_correct += (classes == y).sum().item()
        total_images += len(X)
        end = time.time()
        if progress:
            print(i + 1, '/', num_batches, f'batches trained in {(end - start) / 60:.2f} minutes')

    avg_loss = total_loss / num_batches
    avg_acc = total_correct / total_images

    return avg_loss, avg_acc


def evaluate(dataloader, model, criterion, device, progress=False, parallel=False):
    if parallel:
        assert torch.cuda.device_count() > 1, "Attempted to use multiple GPUs, but only 1 GPU is available"
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.eval()
    num_batches = len(dataloader)

    total_loss = 0
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        start = time.time()
        for i, (X, y) in enumerate(dataloader):
            X, y = X.float().to(device), y.to(device)

            output = model(X)
            total_loss += criterion(output, y).item()
            scores, classes = output.max(dim=1)
            total_correct += (classes == y).sum().item()
            total_images += len(X)
            end = time.time()
            if progress:
                print(i + 1, '/', num_batches, f'batches evaluated in {(end - start) / 60:.2f} minutes')

    avg_loss = total_loss / num_batches
    avg_acc = total_correct / total_images

    return avg_loss


def run(data_path,
        model,
        name,
        device,
        epochs=10,
        lr=0.001,
        bs=16,
        progress=False,
        parallel=False,
        cont=False):
    if cont:
        assert os.path.exists(f'best-{name}-losses.csv'), 'No weights to continue from'
        weights_path = f'best-{name}-weights.pth'
        weights = load(weights_path)
        model.load_state_dict(weights, strict=True)

    if os.path.exists(f'best-{name}-losses.csv'):
        best_val_loss = pd.read_csv(f'best-{name}-losses.csv').iloc[-1]['val loss']
    else:
        best_val_loss = np.inf

    print(f'Initial Validation Loss: {best_val_loss:.5f}')

    val_loader = get_dataloader(data_path, 'val', bs=bs)
    test_loader = get_dataloader(data_path, 'test', bs=bs)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.5,
                                                           patience=1,
                                                           threshold=0.001,
                                                           verbose=True,
                                                           min_lr=1e-5,
                                                           threshold_mode='abs')

    start = time.time()
    metrics = []
    for epoch in range(epochs):
        train_loader = get_dataloader(data_path, 'train', bs=bs)

        train_loss, train_acc = train(train_loader,
                                      model,
                                      criterion,
                                      optimizer,
                                      device,
                                      progress=progress,
                                      parallel=parallel)
        val_loss, val_acc = evaluate(val_loader,
                                     model,
                                     criterion,
                                     device,
                                     progress=progress,
                                     parallel=parallel)

        scheduler.step(val_loss)

        metrics.append([train_loss, train_acc, val_loss, val_acc])
        save(model.state_dict(), f'latest-{name}-weights.pth')

        info = f"Epoch {epoch + 1}: Training Loss: {train_loss:.5f}\t Training Accuracy: {train_acc * 100:.2f}%" \
               f"Validation Loss: {val_loss:.5f}\tValidation Accuracy: {val_acc * 100:.2f}%"

        best = False
        if best_val_loss > val_loss:
            best = True
            torch.save(model.state_dict(), f'best-{name}-weights.pth')
            best_val_loss = val_loss

            test_loss, test_acc = evaluate(test_loader,
                                           model,
                                           criterion,
                                           device,
                                           progress=False,
                                           parallel=parallel)

            info += f"\tCheckpoint!\n\t\t Test Loss: {test_loss:.5f}\tTest Accuracy: {test_acc * 100:.2f}%"

        end = time.time()
        print(info + f'\nTime elapsed: {(end - start) / 60:.2f} minutes')
        if best:
            pd.DataFrame(
                metrics,
                columns=['train loss', 'train acc', 'val loss', 'val acc']
            ).to_csv(
                f'best-{name}-losses.csv',
                index=False
            )
