# scripts/train_lstm.py
"""
=========================
📘 このスクリプトの目的と仕組み（解説）
=========================

🧠 【目的】
- このプログラムは「LSTM + Attention モデル」を用いて、
  各企業の株価を予測するためのモデルを学習します。

📈 【やっていること】
1. 各企業ごとに以下のデータを準備：
   - 過去10営業日分の特徴量（テクニカル指標＋感情スコア）
   - そのデータに基づく「未来5日後・10日後・30日後」の株価（正解）

2. モデルに入力する特徴量：
   - SMA（単純移動平均）5日/15日
   - EMA（指数移動平均）5日
   - RSI（相対力指数）
   - ニュースの感情スコア（その日の平均×記事数）

3. LSTMモデルは「過去10日分の特徴量」から「未来◯営業日後の株価」を予測

4. モデルの精度は RMSE（予測と正解の誤差）で評価

🖼 【出力されるグラフ】
- 青い線：実際の株価（正解データ）
- オレンジの点線：LSTMが予測した株価

イメージ：
         時系列 →
    ┌───────────────────────────┐
    │   実際：青い線             │
    │         ／＼     ／＼      │
    │        ／   ＼／    ＼__   │
    │                            │
    │   予測：オレンジ点線       │
    │       …→………→………→…        │
    └───────────────────────────┘

🚀 【結果】
- 各企業ごとに画像ファイルとして保存されます：
  - 例：data/lstm_predictions/トヨタ_future10.png

"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas.tseries.offsets as offsets

# プロジェクトルートをパスに追加（symbols_config.py 等を読み込むため）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from symbols_config import stock_list

# 日本語フォント設定（Windowsの場合 "Meiryo" など）
rcParams['font.family'] = 'Meiryo'

# ====== 1. LSTM + Attention モデル定義 ======
class StockLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)
        output = self.fc(context_vector)
        return output

# ====== 2. Dataset クラス ======
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 3. シーケンス作成関数（未来n営業日後の予測） ======
def create_sequences(data, target, seq_len, future):
    xs, ys = [], []
    for i in range(len(data) - seq_len - future + 1):
        xs.append(data[i:i+seq_len])
        ys.append(target[i+seq_len+future-1])
    return np.array(xs), np.array(ys)

# ====== 4. 未来予測用の逐次予測関数 ======
def predict_future(model, last_sequence, n_steps, scaler_y, device):
    current_seq = last_sequence.copy()
    predictions = []
    for _ in range(n_steps):
        input_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(input_tensor).cpu().detach().numpy()[0][0]
        predictions.append(pred)
        # ここでは Close 値のみを予測する前提
        new_row = np.array([pred])
        # もし特徴量が1つのみなら new_row のサイズは (1,) とするが current_seq は (seq_len, features)
        # 今回 features = 5 → 株価は1つなので、他の4特徴量は最新値をそのまま維持
        # ここでは簡便のため、new_row を [pred, current_seq[-1,1:]] として横結合
        # ※ 実際は各テクニカル指標の動向を予測するのは難しいため、Close のみ予測し、他は最新値をコピーします。
        new_row = np.hstack(([pred], current_seq[-1, 1:]))
        current_seq = np.vstack((current_seq[1:], new_row))
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_inv = scaler_y.inverse_transform(predictions)
    return predictions_inv.flatten()

# ====== 5. ハイパーパラメータ ======
SEQ_LEN = 10                        # 過去データの日数（タイムステップ数）
FUTURE_STEPS = [5, 10, 30]          # 未来予測対象：5, 10, 30営業日後
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SAVE_DIR = "data/lstm_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 6. CSVデータ読み込み ======
# 「data/merged_sentiment_stock.csv」はニュース・株価などが統合済みのファイルです。
df = pd.read_csv("data/merged_sentiment_stock.csv", parse_dates=["date"])
# 日付を時刻なしのDate型に変換
df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.date
df = df.dropna(subset=["company"])

# ====== 7. 各企業ごとの学習・未来予測 ======
companies = df["company"].unique()
print("対象企業:", companies)

for company in companies:
    print(f"\n🔄 {company} の学習開始...")
    comp_df = df[df["company"] == company].sort_values("date").copy()
    if comp_df.empty:
        print(f"⚠️ {company}: データがありません。スキップ。")
        continue

    # --- 株価からテクニカル指標の再計算 ---
    comp_df["Close"] = pd.to_numeric(comp_df["Close"], errors="coerce")
    comp_df["SMA_5"] = comp_df["Close"].rolling(window=5, min_periods=1).mean()
    comp_df["SMA_15"] = comp_df["Close"].rolling(window=15, min_periods=1).mean()
    comp_df["EMA_5"] = comp_df["Close"].ewm(span=5, adjust=False).mean()
    delta = comp_df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    comp_df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    # --- 同日のニュース感情スコア統合 ---
    # （「score」列はニュースからの感情スコア。 weighted_score = score_mean * 記事数）
    grouped = comp_df.groupby("date").agg({
        "Close": "mean",
        "SMA_5": "mean",
        "SMA_15": "mean",
        "EMA_5": "mean",
        "RSI": "mean",
        "score": ["mean", "count"]
    }).reset_index()
    grouped.columns = ["date", "Close", "SMA_5", "SMA_15", "EMA_5", "RSI", "score_mean", "score_count"]
    grouped["weighted_score"] = grouped["score_mean"] * grouped["score_count"]

    # 日付を datetime 型に変換しておく
    grouped["date"] = pd.to_datetime(grouped["date"])
    # 欠損処理：前方埋め
    grouped[["Close", "SMA_5", "SMA_15", "EMA_5", "RSI", "weighted_score"]] = grouped[["Close", "SMA_5", "SMA_15", "EMA_5", "RSI", "weighted_score"]].ffill()
    
    # 営業日ベースの再構築
    start_date = grouped["date"].min()
    end_date = grouped["date"].max()
    business_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    bdays_df = pd.DataFrame({"date": business_dates})
    merged_df = pd.merge(bdays_df, grouped, on="date", how="left")
    merged_df[["Close", "SMA_5", "SMA_15", "EMA_5", "RSI", "weighted_score"]] = \
    merged_df[["Close", "SMA_5", "SMA_15", "EMA_5", "RSI", "weighted_score"]].ffill().bfill()

    # 日付を object 型ではなく date 型に変換
    merged_df["date"] = merged_df["date"].dt.date

    features = ["SMA_5", "SMA_15", "EMA_5", "RSI", "weighted_score"]
    if merged_df[features + ["Close"]].isnull().any().any() or len(merged_df) < SEQ_LEN + max(FUTURE_STEPS):
        print(f"⚠️ {company}: 欠損またはデータ不足 → スキップ")
        continue

    # スケーリング
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(merged_df[features])
    y_scaled = y_scaler.fit_transform(merged_df[["Close"]])

    # --- 各未来予測ステップごとの学習と評価 ---
    for future in FUTURE_STEPS:
        print(f"▶️ {future}営業日後予測")
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, future)
        if len(X_seq) == 0:
            print(f"⚠️ {company}: シーケンスデータ不足（{future}日後） → スキップ")
            continue

        total_seq = len(X_seq)
        train_size = int(total_seq * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"⚠️ {company}: 学習またはテストデータ不足（{future}日後） → スキップ")
            continue

        train_ds = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StockLSTMAttention(input_size=X_train.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"{company} - Epoch {epoch+1}/{EPOCHS} (future={future}), Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = y_scaler.inverse_transform(y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        print(f"{company} の {future}営業日後 RMSE: {rmse:.2f}")

        # テストグラフ：テストシーケンスに対応する営業日の日付を取得
        test_dates = []
        for i in range(len(X_test)):
            idx = train_size + i + SEQ_LEN + future - 1
            if idx < len(merged_df):
                test_dates.append(merged_df["date"].iloc[idx])
            else:
                test_dates.append(merged_df["date"].iloc[-1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(test_dates, y_test_inv, label="Actual", color="blue")
        plt.plot(test_dates, y_pred_inv, label=f"Predicted ({future}営業日後)", color="orange", linestyle="--")
        plt.title(f"{company} 株価予測 ({future}営業日後) - RMSE: {rmse:.2f}")
        plt.xlabel("日付")
        plt.ylabel("株価")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        test_save_path = os.path.join(SAVE_DIR, f"{company}_future{future}.png")
        plt.savefig(test_save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ {company} → {future}営業日後テスト予測保存: {test_save_path}")

        # --- 完全な未来予測 (最新シーケンスから未知の未来値のみ) ---
        last_sequence = X_scaled[-SEQ_LEN:]
        future_pred = predict_future(model, last_sequence, future, y_scaler, device)
        last_date = pd.to_datetime(merged_df["date"].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=future)
        
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, future_pred, label=f"Future Predicted ({future}営業日後)", color="green", linestyle="--", marker="o")
        plt.title(f"{company} 未来予測 ({future}営業日後)")
        plt.xlabel("日付")
        plt.ylabel("株価")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        future_save_path = os.path.join(SAVE_DIR, f"{company}_future{future}_future.png")
        plt.savefig(future_save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ {company} 未来予測保存: {future_save_path}")
