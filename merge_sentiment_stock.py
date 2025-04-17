import os
import pandas as pd
import yfinance as yf
from datetime import timedelta
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from symbols_config import stock_list

# --- 感情CSV読み込み ---
news_csv = os.path.join("data", "news_sentiment.csv")
try:
    news_df = pd.read_csv(news_csv, parse_dates=["date"])
except Exception as e:
    print(f"❌ ニュースCSV読み込み失敗: {e}")
    sys.exit(1)

news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce").dt.date
news_df = news_df.sort_values("date")

# シンボルマップ作成
name_to_symbol = {}
for s in stock_list:
    name_to_symbol[s["name"]] = s["symbol"] + (".T" if s.get("is_japan") else "")

merged = []
for comp in news_df["company"].unique():
    sym = name_to_symbol.get(comp)
    if not sym:
        print(f"⚠️ {comp}: シンボル未登録 → スキップ")
        continue

    cnews = news_df[news_df["company"] == comp].copy()
    if cnews.empty:
        print(f"⚠️ {comp}: ニュースデータなし → スキップ")
        continue

    # スコア全ZEROなら除外
    if cnews["score"].fillna(0).abs().sum() < 1e-6:
        print(f"⚠️ {comp}: 感情スコア全ゼロ → スキップ")
        continue

    start = cnews["date"].min() - timedelta(days=1)
    end = cnews["date"].max() + timedelta(days=1)
    print(f"📈 {comp} ({sym}) 取得期間: {start}~{end}")

    df_stock = yf.download(sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df_stock.empty:
        print(f"⚠️ {comp}: 株価取得失敗 → スキップ")
        continue

    df_stock.reset_index(inplace=True)
    df_stock["date"] = pd.to_datetime(df_stock["Date"], errors="coerce").dt.date
    df_stock.drop(columns=["Date"], errors="ignore", inplace=True)

    # テクニカル指標
    df_stock["SMA_5"] = df_stock["Close"].rolling(5).mean()
    df_stock["SMA_15"] = df_stock["Close"].rolling(15).mean()
    df_stock["EMA_5"] = df_stock["Close"].ewm(span=5, adjust=False).mean()
    delta = df_stock["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df_stock["RSI"] = 100 - (100 / (1 + avg_gain/avg_loss))

    # マージ
    merged_df = pd.merge(df_stock, cnews, on="date", how="left")
    if merged_df.empty:
        print(f"⚠️ {comp}: マージ失敗 → スキップ")
        continue

    out = os.path.join("data", f"merged_{comp}.csv")
    merged_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ {comp} 保存: {out}")
    merged.append(merged_df)

# 全件統合
if merged:
    all_df = pd.concat(merged, ignore_index=True)
    all_df.to_csv(os.path.join("data", "merged_sentiment_stock.csv"),
                  index=False, encoding="utf-8-sig")
    print("✅ 全企業マージ完了 → data/merged_sentiment_stock.csv")
else:
    print("⚠️ 対象データなし。")
