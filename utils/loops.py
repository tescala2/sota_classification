import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as ps

from torch import save, load
from sklearn.metrics import top_k_accuracy_score
from .data import get_dataloader
from .get_model import get_model


def train(dataloader, model, criterion, optimizer, device, progress=False, parallel=False):
    if parallel:
        assert torch.cuda.device_count() > 1, "Attempted to use multiple GPUs, but only 1 GPU is available"
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.train()
    num_batches = len(dataloader)

    total_loss = 0
    total_correct_1 = 0
    total_correct_5 = 0
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
        scores, _ = output
        total_correct_1 += top_k_accuracy_score(y, scores, k=1)
        total_correct_5 += top_k_accuracy_score(y, scores, k=5)
        total_images += len(X)
        end = time.time()
        if progress:
            print(i + 1, '/', num_batches, f'batches trained in {(end - start) / 60:.2f} minutes')

    avg_loss = total_loss / num_batches
    avg_acc_1 = total_correct_1 / total_images
    avg_acc_5 = total_correct_5 / total_images

    return avg_loss, avg_acc_1, avg_acc_5


def evaluate(dataloader, model, criterion, device, progress=False, parallel=False):
    if parallel:
        assert torch.cuda.device_count() > 1, "Attempted to use multiple GPUs, but only 1 GPU is available"
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.eval()
    num_batches = len(dataloader)

    total_loss = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_images = 0
    with torch.no_grad():
        start = time.time()
        for i, (X, y) in enumerate(dataloader):
            X, y = X.float().to(device), y.to(device)

            output = model(X)
            total_loss += criterion(output, y).item()
            scores, _ = output
            total_correct_1 += top_k_accuracy_score(y, scores, k=1)
            total_correct_5 += top_k_accuracy_score(y, scores, k=5)
            total_images += len(X)
            end = time.time()
            if progress:
                print(i + 1, '/', num_batches, f'batches evaluated in {(end - start) / 60:.2f} minutes')

    avg_loss = total_loss / num_batches
    avg_acc_1 = total_correct_1 / total_images
    avg_acc_5 = total_correct_5 / total_images

    return avg_loss, avg_acc_1, avg_acc_5


def run(data_path,
        model_name,
        device,
        epochs=10,
        lr=0.001,
        bs=64,
        progress=False,
        parallel=False,
        cont=False,
        num_classes=50):
    print(f'Beginning Training on {model_name} for {epochs} epochs')
    model = get_model(num_classes=num_classes, model=model_name)
    if cont:
        print('Loading Weights from Checkpoint')
        assert os.path.exists(f'checkpoint/{model_name}/best-weights.csv'), 'No weights to continue from'
        weights_path = f'checkpoint/{model_name}/best-weights.pth'
        weights = load(weights_path)
        model.load_state_dict(weights, strict=True)

    if os.path.exists(f'checkpoint/{model_name}/best-metrics.csv'):
        best_val_loss = pd.read_csv(f'checkpoint/{model_name}/best-metrics.csv').iloc[-1]['val loss']
    else:
        best_val_loss = np.inf

    print(f'Initial Validation Loss: {best_val_loss:.5f}')

    val_loader = get_dataloader(data_path, 'val', bs=bs, shuffle=False)
    test_loader = get_dataloader(data_path, 'test', bs=bs, shuffle=False)

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

        train_loss, train_acc_1, train_acc_5 = train(train_loader,
                                                     model,
                                                     criterion,
                                                     optimizer,
                                                     device,
                                                     progress=progress,
                                                     parallel=parallel)
        val_loss, val_acc_1, val_acc_5 = evaluate(val_loader,
                                                  model,
                                                  criterion,
                                                  device,
                                                  progress=progress,
                                                  parallel=parallel)

        scheduler.step(val_loss)

        metrics.append([train_loss, val_loss, train_acc_1, val_acc_1, train_acc_5, val_acc_5])
        save(model.state_dict(), f'checkpoint/{model_name}/latest-weights.pth')

        info = f"Epoch {epoch + 1}: Training Loss: {train_loss:.5f}\t Training Accuracy @ 1: {train_acc_1 * 100:.2f}%" \
               f"Validation Loss: {val_loss:.5f}\tValidation Accuracy @ 1: {val_acc_1 * 100:.2f}%"

        best = False
        if best_val_loss > val_loss:
            best = True
            torch.save(model.state_dict(), f'{model_name}/best-weights.pth')
            best_val_loss = val_loss

            test_loss, test_acc_1, test_acc_5 = evaluate(test_loader,
                                                         model,
                                                         criterion,
                                                         device,
                                                         progress=False,
                                                         parallel=parallel)

            info += f"\tCheckpoint!\n\t\t Test Loss: {test_loss:.5f}\tTest Accuracy @ 1: {test_acc_1 * 100:.2f}%"

        end = time.time()
        print(info + f'\nTime elapsed: {(end - start) / 60:.2f} minutes')
        if best:
            pd.DataFrame(
                metrics,
                columns=['train loss', 'val loss', 'train acc @ 1', 'val acc @ 1', 'train acc @ 5', 'val acc @ 5']
            ).to_csv(
                f'checkpoint/{model_name}/best-metrics.csv',
                index=False
            )


def run_inference(data_path,
                  model_name,
                  device,
                  bs=64,
                  progress=False,
                  parallel=False,
                  num_classes=50):
    assert os.path.exists(f'checkpoint/{model_name}/best-weights.csv'), 'No weights to continue from'
    weights_path = f'checkpoint/{model_name}/best-weights.pth'
    weights = load(weights_path)
    model = get_model(num_classes=num_classes, model=model_name)
    model.load_state_dict(weights, strict=True)

    val_loader = get_dataloader(data_path, 'val', bs=bs, shuffle=False)
    test_loader = get_dataloader(data_path, 'test', bs=bs, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc_1, val_acc_5 = evaluate(val_loader,
                                              model,
                                              criterion,
                                              device,
                                              progress=progress,
                                              parallel=parallel)
    test_loss, test_acc_1, test_acc_5 = evaluate(test_loader,
                                                 model,
                                                 criterion,
                                                 device,
                                                 progress=progress,
                                                 parallel=parallel)

    print('Validation Metrics')
    print(
        f"Validation Loss: {val_loss:.5f}\tValidation Accuracy @ 1: {val_acc_1 * 100:.2f}%\tValidation Accuracy @ 5: {val_acc_5 * 100:.2f}%")
    print('Test Metrics')
    print(
        f"Test Loss: {test_loss:.5f}\tTest Accuracy @ 1: {test_acc_1 * 100:.2f}%\tTest Accuracy @ 5: {test_acc_5 * 100:.2f}%")
