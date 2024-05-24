import torch
import os
import numpy as np
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import image_gradients
from torch.nn import L1Loss
from DepthEstimation.dataset import DepthEstimationDataset, return_train_dataloader, return_validation_dataloader
from DepthEstimation.model import Model

DEVICE = torch.device('mps') if torch.backends.mps.is_available() is True else torch.device('cpu')
model = Model().to(DEVICE)


def save_checkpoint(model_state, filename='checkpoint.pth'):
    filename = os.path.join("../checkpoints", filename)
    torch.save(model_state, filename)


def get_loss(y_true, y_predicted, alpha, beta):
    y_true_dx, y_true_dy = image_gradients(y_true)
    y_predicted_dx, y_predicted_dy = image_gradients(y_predicted)

    l1_loss = L1Loss()
    loss = l1_loss(y_true, y_predicted)

    ssim_factor_loss = StructuralSimilarityIndexMeasure().to(DEVICE)
    ssim_loss = 1 - ssim_factor_loss(y_true, y_predicted)
    ssim_loss = ssim_loss.to(DEVICE)
    ssim_loss = torch.clamp(ssim_loss, 0, 1)

    dx_l1_loss = l1_loss(y_true_dx, y_predicted_dx)
    dy_l1_loss = l1_loss(y_true_dy, y_predicted_dy)
    return ssim_loss * (alpha * loss + beta * (dx_l1_loss + dy_l1_loss))


def train_valid_routine(batch_size, epochs, learning_rate, resume_training=True, alpha=0.7, beta=0.3):
    train_dataloader = return_train_dataloader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-3)
    train_losses = np.array(list())

    best_loss, start_epoch = np.inf, 0
    checkpoint_path = os.path.join('../checkpoints', 'checkpoint.pth')
    if resume_training is True and os.path.exists(checkpoint_path) is True:
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']

    for epoch in range(epochs):
        if start_epoch != 0 and epoch <= start_epoch:
            print(f'[DepthEstimation] Skip training on epoch {epoch + 1}/{epochs}')
            continue

        model.train()
        epoch_train_losses = np.array(list())

        print(f'[DepthEstimation] Trining epoch {epoch + 1}/{epochs} started.')
        for i, (train_image, train_depth_map) in tqdm(enumerate(train_dataloader)):
            print(f'[DepthEstimation] Processing image {i + 1}/{len(train_dataloader)} for training')
            optimizer.zero_grad()

            if torch.backends.mps.is_available():
                train_image, train_depth_map = train_image.to(DEVICE), train_depth_map.to(DEVICE)
            depth_map_prediction = model(train_image)
            train_loss = get_loss(train_depth_map, depth_map_prediction, alpha, beta)

            optimizer.step()
            train_loss.backward()

            np.append(epoch_train_losses, train_loss.item())

        epoch_loss = epoch_train_losses.mean()
        np.append(train_losses, epoch_loss)
        print(f'[DepthEstimation] Train loss for epoch {epoch + 1}/{epochs} is: {epoch_loss}')

        print(f'[DepthEstimation] Save the model after training current epoch.')
        save_checkpoint(dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_loss=best_loss
        ))

    valid_dataloader = return_validation_dataloader(batch_size)
    model.eval()
    for epoch in range(epochs):
        epoch_valid_losses = np.array(list())
        print(f'[DepthEstimation] Validation epoch {epoch + 1}/{epochs} started.')
        for i, (valid_image, valid_depth_map) in tqdm(enumerate(valid_dataloader)):
            if torch.backends.mps.is_available():
                valid_image, valid_depth_map = valid_image.to(DEVICE), valid_depth_map.to(DEVICE)

            with torch.no_grad():
                depth_map_prediction = model(valid_image)

            valid_loss = get_loss(valid_depth_map, depth_map_prediction, alpha, beta)
            np.append(epoch_valid_losses, valid_loss.item())

        epoch_loss = epoch_valid_losses.mean()
        print(f'[DepthEstimation] Validation loss for epoch {epoch + 1}/{epochs} is: {epoch_loss}')

        if epoch_loss < best_loss:
            print(f'[DepthEstimation] Save the model after validation with epoch_loss: {epoch_loss}/{best_loss}.')
            save_checkpoint(dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                best_loss=best_loss
            ))
