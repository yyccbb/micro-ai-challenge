import torch
import torch.nn as nn

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y, padding_value=0.0):
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

def create_data_loaders(x1, x2, y, train_ratio=0.8, val_ratio=0.1, batch_size=32, shuffle=False, num_workers=0, padding_value=0.0, random_state=42):
    dataset = TimeSeriesDataset(x1, x2, y, padding_value)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_state)
    )

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
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # TODO: Try AdamW, try weight_decay=1e-5
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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

    model.load_state_dict(torch.load(model_save_path))

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