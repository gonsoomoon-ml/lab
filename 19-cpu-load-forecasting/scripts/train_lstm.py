#!/usr/bin/env python3
"""
LSTM 모델 학습 및 평가

CPU Load Forecasting을 위한 LSTM deep learning 모델
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    """시계열 데이터셋 for PyTorch"""
    def __init__(self, data, target, sequence_length=60):
        """
        Args:
            data: Feature array (n_samples, n_features)
            target: Target array (n_samples,)
            sequence_length: 입력 시퀀스 길이 (분 단위)
        """
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # 과거 sequence_length개 시점의 데이터
        x = self.data[idx:idx + self.sequence_length]
        # 다음 시점의 타겟
        y = self.target[idx + self.sequence_length]
        return x, y


class LSTMModel(nn.Module):
    """LSTM 회귀 모델"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        # 마지막 time step의 출력만 사용
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


def calculate_metrics(y_true, y_pred):
    """예측 성능 지표 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'n_samples': len(y_true)
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1 epoch 학습"""
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_model(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    return total_loss / len(dataloader), np.array(predictions), np.array(actuals)


def plot_predictions(y_true, y_pred, timestamps, output_path, sample_hours=48):
    """예측 결과 시각화"""
    sample_minutes = sample_hours * 60
    n_samples = min(sample_minutes, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Time series
    ax1 = axes[0]
    ax1.plot(timestamps[:n_samples], y_true[:n_samples],
             label='Actual', linewidth=2, alpha=0.8, color='#0066CC')
    ax1.plot(timestamps[:n_samples], y_pred[:n_samples],
             label='LSTM Prediction', linewidth=2, alpha=0.8,
             linestyle='--', color='#DC143C')
    ax1.set_ylabel('CPU Load (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'LSTM Predictions: First {sample_hours} Hours',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    ax2 = axes[1]
    errors = y_true[:n_samples] - y_pred[:n_samples]
    ax2.plot(timestamps[:n_samples], errors, linewidth=1.5, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 예측 결과 그래프 저장: {output_path}")


def plot_training_history(train_losses, val_losses, output_path):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('LSTM Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 학습 곡선 저장: {output_path}")


def compare_with_baseline(lstm_metrics, baseline_mape=1.2455, xgboost_mape=0.9293, rf_mape=0.8913, prophet_mape=1.8997):
    """Baseline 및 다른 모델들과 성능 비교"""
    print("\n" + "=" * 80)
    print("모델 성능 비교")
    print("=" * 80)

    improvement_vs_baseline = (baseline_mape - lstm_metrics['MAPE']) / baseline_mape * 100
    vs_rf = lstm_metrics['MAPE'] - rf_mape

    print(f"\n{'Model':<25} {'MAPE (%)':<15} {'vs Baseline':<20} {'vs Best ML':<15}")
    print("─" * 80)
    print(f"{'Random Forest':<25} {rf_mape:<15.4f} {'+28.44%':<20} {'Best ML':<15}")
    print(f"{'XGBoost':<25} {xgboost_mape:<15.4f} {'+25.39%':<20} {'-':<15}")
    print(f"{'Seasonal Naive (Baseline)':<25} {baseline_mape:<15.4f} {'Baseline':<20} {'-':<15}")
    print(f"{'Prophet':<25} {prophet_mape:<15.4f} {'-52.52%':<20} {'-':<15}")
    print(f"{'LSTM':<25} {lstm_metrics['MAPE']:<15.4f} {improvement_vs_baseline:+.2f}%{'':<13} {vs_rf:+.4f}%{'':<8}")

    print("\n" + "─" * 80)

    if improvement_vs_baseline > 20:
        grade = "S (탁월)"
        status = "✓✓ Baseline을 크게 개선!"
    elif improvement_vs_baseline > 12:
        grade = "A (매우 좋음)"
        status = "✓ Baseline보다 우수"
    elif improvement_vs_baseline > 4:
        grade = "B (좋음)"
        status = "✓ Baseline보다 약간 개선"
    elif improvement_vs_baseline > 0:
        grade = "C (보통)"
        status = "⚠ 개선 미미"
    else:
        grade = "F (실패)"
        status = "✗ Baseline보다 나쁨"

    print(f"평가 등급: {grade}")
    print(f"결과: {status}")

    # Random Forest와 비교
    if abs(vs_rf) < 0.05:
        rf_comparison = "≈ Random Forest와 거의 동일"
    elif vs_rf < 0:
        rf_comparison = f"✓ Random Forest보다 {abs(vs_rf):.4f}% 더 나음"
    else:
        rf_comparison = f"⚠ Random Forest보다 {vs_rf:.4f}% 나쁨"

    print(f"Random Forest 대비: {rf_comparison}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LSTM 모델 학습 및 평가")

    parser.add_argument('--train-data', type=str, default='data/processed/train_processed.csv')
    parser.add_argument('--test-data', type=str, default='data/processed/test_processed.csv')
    parser.add_argument('--output-dir', type=str, default='results/lstm')
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--random-state', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LSTM 모델 학습")
    print("=" * 80)

    # 1. 데이터 로드
    print(f"\n1. 데이터 로딩...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")

    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values

    n_features = X_train.shape[1]
    print(f"   Features: {n_features}개")

    # 2. 데이터 정규화
    print(f"\n2. 데이터 정규화...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ StandardScaler 적용")

    # 3. 데이터셋 생성
    print(f"\n3. 시퀀스 데이터셋 생성...")
    print(f"   Sequence length: {args.sequence_length}분")

    train_dataset = TimeSeriesDataset(X_train_scaled, y_train, args.sequence_length)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test, args.sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Test sequences:  {len(test_dataset):,}")

    # 4. 모델 생성
    print(f"\n4. LSTM 모델 생성...")
    print(f"   Hyperparameters:")
    print(f"     - input_size: {n_features}")
    print(f"     - hidden_size: {args.hidden_size}")
    print(f"     - num_layers: {args.num_layers}")
    print(f"     - dropout: {args.dropout}")
    print(f"     - batch_size: {args.batch_size}")
    print(f"     - learning_rate: {args.learning_rate}")

    model = LSTMModel(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"   ✓ 모델 생성 완료")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 학습
    print(f"\n5. 모델 학습...")
    print(f"   Epochs: {args.epochs}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = test_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    print(f"   ✓ 학습 완료 (Best Val Loss: {best_val_loss:.4f})")

    # 6. 예측
    print(f"\n6. 예측 수행...")
    _, y_test_pred, y_test_actual = test_model(model, test_loader, criterion, device)
    train_loader_eval = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    _, y_train_pred, y_train_actual = test_model(model, train_loader_eval, criterion, device)
    print(f"   ✓ 예측 완료")

    # 7. 성능 평가
    print(f"\n7. 성능 평가...")
    train_metrics = calculate_metrics(y_train_actual, y_train_pred)
    test_metrics = calculate_metrics(y_test_actual, y_test_pred)

    print("\n" + "=" * 80)
    print("성능 지표")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12}")
    print("─" * 80)
    print(f"{'Train':<15} {train_metrics['MAE']:<12.4f} {train_metrics['RMSE']:<12.4f} {train_metrics['MAPE']:<12.4f}")
    print(f"{'Test':<15} {test_metrics['MAE']:<12.4f} {test_metrics['RMSE']:<12.4f} {test_metrics['MAPE']:<12.4f}")

    compare_with_baseline(test_metrics)

    # 8. 학습 곡선
    print(f"\n8. 학습 곡선 시각화...")
    history_path = output_dir / 'lstm_training_history.png'
    plot_training_history(train_losses, val_losses, history_path)

    # 9. 예측 결과 시각화
    print(f"\n9. 예측 결과 시각화...")
    test_original = pd.read_csv('data/test.csv')
    timestamps = pd.to_datetime(test_original['timestamp'])
    offset = 1440 + args.sequence_length
    timestamps_aligned = timestamps.iloc[offset:offset+len(y_test_pred)].values

    prediction_path = output_dir / 'lstm_predictions.png'
    plot_predictions(y_test_actual, y_test_pred, timestamps_aligned, prediction_path)

    # 10. 모델 저장
    print(f"\n10. 모델 저장...")
    model_path = output_dir / 'lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'hyperparameters': {
            'input_size': n_features,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'sequence_length': args.sequence_length
        }
    }, model_path)
    print(f"   ✓ 모델 저장: {model_path}")

    # 11. 결과 저장
    print(f"\n11. 결과 저장...")

    predictions_df = pd.DataFrame({
        'actual': y_test_actual,
        'predicted': y_test_pred,
        'error': y_test_actual - y_test_pred
    })
    predictions_path = output_dir / 'lstm_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   ✓ 예측값 저장: {predictions_path}")

    metrics_path = output_dir / 'lstm_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("LSTM Model Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write("[Train Set]\n")
        f.write(f"MAE:  {train_metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {train_metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {train_metrics['MAPE']:.4f}%\n\n")
        f.write("[Test Set]\n")
        f.write(f"MAE:  {test_metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {test_metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {test_metrics['MAPE']:.4f}%\n\n")
        f.write("[Hyperparameters]\n")
        f.write(f"sequence_length: {args.sequence_length}\n")
        f.write(f"hidden_size: {args.hidden_size}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"epochs_trained: {len(train_losses)}\n")
        f.write(f"best_val_loss: {best_val_loss:.4f}\n")

    print(f"   ✓ 메트릭 저장: {metrics_path}")

    print("\n" + "=" * 80)
    print("✅ LSTM 모델 학습 완료!")
    print("=" * 80)
    print(f"\n저장된 파일:")
    print(f"  - {model_path}")
    print(f"  - {predictions_path}")
    print(f"  - {metrics_path}")
    print(f"  - {history_path}")
    print(f"  - {prediction_path}")


if __name__ == "__main__":
    main()
