import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'  # 日本語フォント設定

# --- 1. データ取得 & 前処理スクリプト実行 ---
def run_step(command, description):
    print(f"\n▶️ {description} 実行中...")
    subprocess.run(["python", command], check=True)
    print(f"✅ {description} 完了\n")

# ニュース収集
run_step("scrape_news.py", "scrape_news.py（ニュース収集）")
# 株価＋感情のマージ
run_step("merge_sentiment_stock.py", "merge_sentiment_stock.py（株価と感情統合）")
# LSTM予測モデル学習
run_step("train_lstm.py", "train_lstm.py（予測モデル学習）")

# --- 2. 感情 vs 株価グラフ生成（全企業） ---
print("📊 各企業の感情スコア vs 株価をプロット中...")

df = pd.read_csv("data/merged_sentiment_stock.csv", parse_dates=["date"])
companies = df["company"].unique()

output_dir = os.path.join("data", "sentiment_vs_stock")
os.makedirs(output_dir, exist_ok=True)

for company in companies:
    company_df = df[df["company"] == company].sort_values("date")
    if company_df.empty or "score" not in company_df.columns:
        print(f"⚠️ {company}：データ不足またはスコアなし → スキップ")
        continue

    # 図と軸の設定
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(company_df["date"], company_df["Close"], label="株価", color="blue")
    ax1.set_xlabel("日付")
    ax1.set_ylabel("株価", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # 第二軸
    ax2 = ax1.twinx()
    ax2.plot(company_df["date"], company_df["score"], label="感情スコア", color="red", alpha=0.5)
    ax2.set_ylabel("感情スコア", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # レイアウト調整
    plt.title(f"{company} - 感情スコア vs 株価")
    ax1.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()

    # 保存
    out_path = os.path.join(output_dir, f"{company}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {company} の感情vs株価グラフ 保存: {out_path}")

# --- 3. Streamlit アプリ起動 ---
print("\n🚀 Streamlit アプリを起動中...")
subprocess.run(["streamlit", "run", "app.py"])
