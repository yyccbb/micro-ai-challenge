import os
from joblib import dump, load
import torch
import torch.nn as nn
from math import sqrt

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y, lengths1, lengths2, padding_value=0.0):
        """
        Args:
            x1: (4140, 755, 20) tensor
            x2: (4140, 755, 45) tensor
            y: (4140, 49) tensor
            padding_value: Value used for padding (default: 0.0)
        """
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.padding_value = padding_value
        assert len(x1) == len(x2) == len(y), "All inputs must have the same number of samples!"
        self.lengths1 = get_sequence_lengths(x1, padding_value)
        self.lengths2 = get_sequence_lengths(x2, padding_value)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        return (
            self.x1[item],
            self.x2[item],
            self.y[item],
            self.lengths1[item],
            self.lengths2[item]
        )

def get_sequence_lengths(x, padding_value=0.0):
    mask = (x != padding_value).any(dim=-1)
    reversed_mask = torch.flip(mask, dims=[1])
    lengths = mask.size(1) - reversed_mask.int().argmax(dim=1)
    return lengths.to(dtype=torch.long, device=x.device)

def create_mask_from_lengths(lengths, max_length):
    batch_size = lengths.shape[0]
    range_tensor = torch.arange(max_length, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return range_tensor < lengths.unsqueeze(1)

def compute_standardization_stats(x, mask=None, dim=(0, 1), epsilon=1e-8):
    if mask is not None:
        float_mask = mask.float()
        while float_mask.dim() < x.dim():
            float_mask = float_mask.unsqueeze(-1)

        x_sum = torch.sum((float_mask * x), dim=dim, keepdim=True)
        count = torch.sum(float_mask, dim=dim, keepdim=True) # TODO: Check whether division by 0 could occur
        x_mean = x_sum / count

        x_var = torch.sum(((x - x_mean) * float_mask) ** 2, dim=dim, keepdim=True) / (count - 1).clamp(min=1)
        x_std = torch.sqrt(x_var + epsilon)
    else:
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_std = torch.std(x, dim=dim, keepdim=True).clamp(min=sqrt(epsilon))

    return x_mean, x_std

def standardize_data(x, x_mean, x_std, mask=None, padding_value=0.0):
    standardized = (x - x_mean) / x_std
    if mask is not None:
        expanded_mask = mask
        while expanded_mask.dim() < standardized.dim():
            expanded_mask = expanded_mask.unsqueeze(-1)
        standardized = torch.where(expanded_mask, standardized, padding_value)

    return standardized

def create_data_loaders(x1, x2, y, train_ratio=0.8, val_ratio=0.1, batch_size=32, shuffle=False, standardize=True, num_workers=0, padding_value=0.0, random_state=42):
    total_size = len(x1)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Create indices for splits
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(random_state))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    lengths1 = get_sequence_lengths(x1, padding_value)
    lengths2 = get_sequence_lengths(x2, padding_value)

    if standardize:
        x1_train = x1[train_indices]
        x2_train = x2[train_indices]
        # TODO: Test whether need to standardize y

        lengths1_train = get_sequence_lengths(x1_train, padding_value)
        lengths2_train = get_sequence_lengths(x2_train, padding_value)

        mask1_train = create_mask_from_lengths(lengths1_train, x1_train.size(1))
        mask2_train = create_mask_from_lengths(lengths2_train, x2_train.size(1))

        x1_mean, x1_std = compute_standardization_stats(x1_train, mask1_train, dim=(0, 1))
        x2_mean, x2_std = compute_standardization_stats(x2_train, mask2_train, dim=(0, 1))

        mask1 = create_mask_from_lengths(lengths1, x1.size(1))
        mask2 = create_mask_from_lengths(lengths2, x2.size(1))

        x1 = standardize_data(x1, x1_mean, x1_std, mask1, padding_value)
        x2 = standardize_data(x2, x2_mean, x2_std, mask2, padding_value)

    dataset = TimeSeriesDataset(x1, x2, y, lengths1, lengths2, padding_value)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True) # TODO: Try pin_memory=False

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_data in train_loader:
        x1, x2, y, lengths1, lengths2 = batch_data
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lengths1, lengths2 = lengths1.to(device), lengths2.to(device)

        optimizer.zero_grad()
        outputs = model(x1, x2, lengths1, lengths2)
        loss = criterion(outputs, y)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # TODO: See if need this

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            x1, x2, y, lengths1, lengths2 = batch_data
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            lengths1, lengths2 = lengths1.to(device), lengths2.to(device)

            outputs = model(x1, x2, lengths1, lengths2)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def train_model(model, train_loader, val_loader, num_epochs=40, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu', patience=10, min_delta=1e-4, model_save_path="best_model.pth"
):
    print(f"device: {device}")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # TODO: Try AdamW, try weight_decay=1e-5
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    ) # TODO: Try different hyperparameters

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        lr_scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Printing
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model weights
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

        print("-" * 20)

    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    return train_losses, val_losses

def test_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_losses = []

    criterion = nn.MSELoss()

    print(f"Testing model on {len(test_loader)} batches...")
    with torch.no_grad():
        for batch_data in test_loader:
            x1, x2, y, lengths1, lengths2 = batch_data
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            lengths1, lengths2 = lengths1.to(device), lengths2.to(device)

            outputs = model(x1, x2, lengths1, lengths2)
            loss = criterion(outputs, y)
            all_losses.append(loss.item())
            all_predictions.append(outputs.cpu())
            all_targets.append(y.cpu())

    test_loss = sum(all_losses) / len(all_losses)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    results = {
        'test_loss': test_loss,
        'predictions': all_predictions,
        'targets': all_targets
    }

    # Regression metrics
    mse = torch.nn.functional.mse_loss(all_predictions, all_targets).item()
    mae = torch.nn.functional.l1_loss(all_predictions, all_targets).item()

    # R^2 score (coefficient of determination)
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    rmse = torch.sqrt(torch.tensor(mse)).item()

    results.update({
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2_score.item()
    })

    return results


def train_and_predict_with_validation(
        model,
        x1, x2, y,
        test_x1, test_x2,
        num_epochs=100,  # Increased default for early stopping
        learning_rate=1e-3,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        padding_value=0.0,
        num_workers=0,
        random_state=42,
        # --- Early stopping and Scheduler parameters ---
        patience=10,
        min_delta=1e-4,
        model_save_path="best_final_model.pth"
):
    """
    Splits data into train/val, trains with early stopping, and predicts on a test set.

    This function now:
    1. Splits the provided data (x1, x2, y) into a 90% training and 10% validation set.
    2. Computes standardization stats ONLY from the 90% training portion.
    3. Applies this standardization to the train, validation, and final test sets.
    4. Trains the model using the train set, while monitoring loss on the validation set
       for early stopping and for a ReduceLROnPlateau learning rate scheduler.
    5. Saves the best performing model weights based on validation loss.
    6. Loads the best model and uses it to predict on the separate test_x1, test_x2.

    Args:
        ... (previous args) ...
        patience (int): How many epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in val_loss to be considered an improvement.
        scheduler_patience (int): Patience for the ReduceLROnPlateau scheduler.
        scheduler_factor (float): Factor by which the LR is reduced by the scheduler.
        model_save_path (str): Path to save the temporary best model weights.

    Returns:
        torch.Tensor: The predictions for the test set.
    """
    print("--- Starting Training with Validation Split and Prediction ---")
    model.to(device)

    # 1. Split data into 90% training and 10% validation
    print("Step 1: Splitting data into 90% train and 10% validation sets...")
    total_size = len(x1)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(random_state))
    val_size = int(0.1 * total_size)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    lengths1 = get_sequence_lengths(x1, padding_value)
    lengths2 = get_sequence_lengths(x2, padding_value)

    x1_train = x1[train_indices]
    x2_train = x2[train_indices]

    lengths1_train = get_sequence_lengths(x1_train, padding_value)
    lengths2_train = get_sequence_lengths(x2_train, padding_value)

    mask1_train = create_mask_from_lengths(lengths1_train, x1_train.size(1))
    mask2_train = create_mask_from_lengths(lengths2_train, x2_train.size(1))

    x1_mean, x1_std = compute_standardization_stats(x1_train, mask1_train, dim=(0, 1))
    x2_mean, x2_std = compute_standardization_stats(x2_train, mask2_train, dim=(0, 1))

    mask1 = create_mask_from_lengths(lengths1, x1.size(1))
    mask2 = create_mask_from_lengths(lengths2, x2.size(1))

    x1 = standardize_data(x1, x1_mean, x1_std, mask1, padding_value)
    x2 = standardize_data(x2, x2_mean, x2_std, mask2, padding_value)

    dataset = TimeSeriesDataset(x1, x2, y, lengths1, lengths2, padding_value)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print(f"Using Adam optimizer and ReduceLROnPlateau scheduler.")

    # 5. Train the model with early stopping
    print(f"\nStep 3: Training model with validation for up to {num_epochs} epochs...")
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")


        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved to {best_val_loss:.6f}. Saving model to {model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("Training complete.")

    # 6. Load best model and predict on the final test set
    print(f"\nStep 4: Loading best model from {model_save_path} and generating predictions...")
    if not os.path.exists(model_save_path):
        print(
            "Warning: No model was saved. This may happen if training was too short or no improvement was found. Using the model from the last epoch.")
    else:
        model.load_state_dict(torch.load(model_save_path))

    # Standardize the final test data using the stats from the training split
    test_lengths1 = get_sequence_lengths(test_x1, padding_value)
    test_lengths2 = get_sequence_lengths(test_x2, padding_value)
    test_mask1 = create_mask_from_lengths(test_lengths1, test_x1.size(1))
    test_mask2 = create_mask_from_lengths(test_lengths2, test_x2.size(1))
    test_x1_std = standardize_data(test_x1, x1_mean, x1_std, test_mask1, padding_value)
    test_x2_std = standardize_data(test_x2, x2_mean, x2_std, test_mask2, padding_value)

    model.eval()
    pred_y = None
    with torch.no_grad():
        test_x1_dev = test_x1_std.to(device)
        test_x2_dev = test_x2_std.to(device)
        test_lengths1_dev = test_lengths1.to(device)
        test_lengths2_dev = test_lengths2.to(device)
        pred_y = model(test_x1_dev, test_x2_dev, test_lengths1_dev, test_lengths2_dev)
        pred_y = pred_y.cpu()

    dump(pred_y, 'prediction/predicted_y.joblib')
    print("Prediction generation complete.")
    print("--- Process Finished ---")
    return pred_y