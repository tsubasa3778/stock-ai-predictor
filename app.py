import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import time
import altair as alt
from scrape_news import run_scraper

# ─── ページ設定 ─────────────────────────
st.set_page_config(page_title="株価予測AIアプリ", layout="wide")
matplotlib.rcParams['font.family'] = 'Meiryo'

# ─── モバイル用CSS＆コードスクロール ─────────────────────────
st.markdown("""
<style>
@media only screen and (max-width: 600px) {
    .main { padding: 1rem; }
    h1 { font-size: 1.5rem; }
    .block-container { padding: 0.5rem 1rem; }
}
/* コードブロック横スクロール */
.streamlit-expanderContent pre {
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)
st.title("自分専用 株価予測 AI アプリ 📈")

# ─── ディレクトリ検出 ───────────────────
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
project_data = os.path.join(project_root, "data")
app_data = os.path.join(script_dir, "data")
DATA_DIR = project_data if os.path.exists(project_data) else app_data

# ─── ヘルパー関数 ─────────────────────────
def find_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p): return p
    return None

# ─── 企業リスト読み込み ───────────────────
try:
    from symbols_config import stock_list
    companies = [s["name"] for s in stock_list]
except:
    companies = []

# ─── セッションステート初期化 ─────────────
if "scraping" not in st.session_state: st.session_state.scraping = False
if "stop_scraping" not in st.session_state: st.session_state.stop_scraping = False

# ─── サイドメニュー ───────────────────────
st.sidebar.title("メニュー")
page = st.sidebar.selectbox("ページ選択", [
    "ニュース感情分析",
    "LSTM株価予測",
    "コード閲覧",
    "感情 vs 株価変化"
])

# ====================
# ニュース感情分析 ページ
# ====================
if page == "ニュース感情分析":
    st.header("📰 ニュース感情分析データ")

    # CSV 検出
    path1 = os.path.join(DATA_DIR, "news_sentiment.csv")
    path2 = os.path.join(DATA_DIR, "news_sentiment", "news_sentiment.csv")
    csv_path = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)

    if csv_path:
        df_all = pd.read_csv(csv_path, parse_dates=["date"])
        df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)

        sent_companies = sorted(df_all["company"].unique())
        selected_company = st.selectbox("企業を選択してください", sent_companies)
        df = df_all[df_all["company"] == selected_company]

        st.subheader("ニュース記事一覧")
        st.dataframe(df.tail(500), use_container_width=True)

        months = sorted(df["year_month"].unique(), reverse=True)
        selected_month = st.selectbox("月を選択してください", months)
        df_month = df[df["year_month"] == selected_month]

        # 日別平均スコア + 7日移動平均
        daily = df_month.set_index("date")["score"] \
                        .resample("D").mean().fillna(0).to_frame() \
                        .rename(columns={"score": "score"})
        daily["ma7"] = daily["score"].rolling(7, center=True).mean() \
                               .fillna(method="ffill").fillna(method="bfill")
        daily = daily.reset_index()

        st.subheader(f"{selected_month} の感情スコア日別推移")
        base = alt.Chart(daily).encode(x=alt.X("date:T", title="日付"))
        points = base.mark_circle(size=40).encode(
            y=alt.Y("score:Q", title="スコア"),
            color=alt.condition("datum.score >= datum.ma7", alt.value("steelblue"), alt.value("orange")),
            tooltip=[
                alt.Tooltip(field="date", type="temporal", title="日付"),
                alt.Tooltip(field="score", type="quantitative", format=".3f", title="スコア"),
                alt.Tooltip(field="ma7", type="quantitative", format=".3f", title="7日移動平均")
            ]
        )
        line = base.mark_line(size=3).encode(
            y=alt.Y("ma7:Q", title="7日移動平均"),
            tooltip=[
                alt.Tooltip(field="date", type="temporal", title="日付"),
                alt.Tooltip(field="ma7", type="quantitative", format=".3f", title="7日移動平均")
            ]
        )
        st.altair_chart(alt.layer(points, line).properties(height=300), use_container_width=True)

        # 月別記事件数
        monthly = df.set_index("date")["title"].resample("M").count().reset_index().rename(columns={"title":"件数"})
        st.subheader("月別記事件数（過去1年）")
        bar = alt.Chart(monthly).mark_bar().encode(
            x=alt.X("date:T", timeUnit="yearmonth", title="年月"),
            y=alt.Y("件数:Q", title="件数"),
            tooltip=[
                alt.Tooltip(field="date", type="temporal", timeUnit="yearmonth", title="年月"),
                alt.Tooltip(field="件数", type="quantitative", title="件数")
            ]
        ).properties(height=200)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("まだニュースデータがありません。下のボタンで収集してください。")

    # スクレイピング
    if not st.session_state.scraping and st.button("📥 ニュースを収集する"):
        st.session_state.scraping = True
        st.session_state.stop_scraping = False
    if st.session_state.scraping:
        pb = st.progress(0); status = st.empty()
        if st.button("🛑 キャンセル"): st.session_state.stop_scraping = True
        start = time.time()
        def cb(r):
            if st.session_state.stop_scraping: raise RuntimeError("🛑 キャンセルされました。")
            elapsed = time.time() - start
            eta = (elapsed/r)*(1-r) if r>0 else 0
            m,s = divmod(int(eta),60)
            status.text(f"進捗：{r*100:.1f}% | 残り：{m}分{s}秒")
            pb.progress(min(r,1.0))
        try:
            run_scraper(progress_callback=cb, should_stop=lambda: st.session_state.stop_scraping)
            st.success("✅ 完了！")
        except RuntimeError as e:
            st.warning(str(e))
        st.session_state.scraping = False
        st.session_state.stop_scraping = False

# ====================
# LSTM株価予測 ページ
# ====================
elif page == "LSTM株価予測":
    st.header("📊 株価予測結果")
    company = st.selectbox("企業を選択してください", companies)
    future_path = os.path.join(DATA_DIR, "lstm_predictions", f"{company}_future10.png")

    if os.path.exists(future_path):
        st.image(future_path, caption=f"{company} - 10営業日後予測", use_container_width=True)
    else:
        st.warning(f"{company} の予測画像が見つかりません。")


# ====================
# コード閲覧 ページ
# ====================
elif page == "コード閲覧":
    st.header("📄 アプリコード一覧")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    files = ["scrape_news.py","merge_sentiment_stock.py","train_lstm.py","app.py","symbols_config.py"]
    for fname in files:
        file_path = os.path.join(base_dir, fname)
        if os.path.exists(file_path):
            st.subheader(fname)
            code = open(file_path, encoding="utf-8").read()
            with st.expander("コードを表示"): st.code(code, language="python")
            st.download_button(label=f"{fname} をダウンロード", data=code, file_name=fname, mime="text/x-python")
        else:
            st.warning(f"{fname} が見つかりません。")

# ====================
# 感情 vs 株価変化 ページ
# ====================
elif page == "感情 vs 株価変化":
    st.header("💬 感情スコア vs 株価変化")
    sel = st.selectbox("企業を選択してください", companies)
    vs_img = find_path(
        os.path.join(project_data, "sentiment_vs_stock", f"{sel}.png"),
        os.path.join(app_data,     "sentiment_vs_stock", f"{sel}.png")
    )
    if vs_img:
        st.image(Image.open(vs_img), caption="感情と株価の比較", use_container_width=True)
    else:
        st.warning(f"{sel} の比較グラフが見つかりません。")

st.markdown("---")
st.markdown("📌 **予測誤差（RMSE）**：予測と実際の株価のズレを示す指標。小さいほど高精度。")

import subprocess
import threading
import time

# --- Cloudflare Tunnel 起動関数 ---
def start_tunnel():
    time.sleep(5)  # Streamlitサーバー起動待ち
    try:
        result = subprocess.run(
            ["cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("✅ Cloudflare Tunnel 終了")
        print(result.stdout)
    except Exception as e:
        print("❌ Tunnel 起動エラー:", e)

# --- 別スレッドでトンネル起動（Streamlitと並行実行） ---
threading.Thread(target=start_tunnel, daemon=True).start()
