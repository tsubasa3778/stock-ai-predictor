import os
import sys
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from transformers import pipeline

# symbols_config 読み込み
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from symbols_config import stock_list

def run_scraper(progress_callback=None, should_stop=lambda: False):
    """
    Yahoo!ニュースをスクレイピングし、BERTで感情分析。
    結果を data/news_sentiment.csv に保存。
    progress_callback(ratio: float) で進捗通知。
    should_stop(): bool でキャンセルチェック。
    """
    model_name = "christian-phu/bert-finetuned-japanese-sentiment"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=0)

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365)
    total_days = (end_date - start_date).days + 1
    total_iterations = len(stock_list) * total_days
    iteration = 0

    all_results = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for stock in stock_list:
        if should_stop():
            return
        company_name = stock["name"]

        for day_offset in range(total_days):
            if should_stop():
                return

            date_obj = start_date + timedelta(days=day_offset)
            ymd = date_obj.strftime("%Y-%m-%d")
            query = f"{company_name} 株価"
            search_url = f"https://news.yahoo.co.jp/search/?p={query}&fd={ymd}&td={ymd}"

            iteration += 1
            if progress_callback:
                progress_callback(iteration / total_iterations)

            try:
                res = requests.get(search_url, headers=headers, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                links = soup.find_all(
                    "a",
                    href=lambda x: x and x.startswith("https://news.yahoo.co.jp/articles/")
                )
                article_urls = list(dict.fromkeys(link["href"] for link in links))
            except Exception:
                continue

            for url in article_urls:
                if should_stop():
                    return

                try:
                    article = requests.get(url, headers=headers, timeout=10)
                    article_soup = BeautifulSoup(article.content, "html.parser")
                    title_tag = article_soup.find("h1")
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"

                    # 記事本文抽出と不要テキストの除去
                    paragraphs = article_soup.select("article p, .article-body p, .main-content p, p")
                    # JavaScript無効メッセージを含む段落をフィルタリング
                    filtered = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if not text:
                            continue
                        if "JavaScriptが無効" in text:
                            continue
                        filtered.append(text)
                    body = "\n".join(filtered)
                except Exception:
                    continue

                try:
                    text = body if body.strip() else title
                    result = sentiment_analyzer(text[:512])[0]
                    label = result["label"]
                    score = float(result["score"])
                except Exception:
                    label = "NEUTRAL"
                    score = 0.5

                all_results.append({
                    "date": ymd,
                    "company": company_name,
                    "title": title,
                    "body": body,
                    "label": label,
                    "score": score,
                    "url": url
                })

    # 保存
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/news_sentiment.csv", index=False, encoding="utf-8-sig")


# 単体実行用
if __name__ == "__main__":
    print("📄 ニュース収集開始...")
    
    def console_progress(r):
        pct = f"{r*100:5.1f}%"
        print(f"\r進捗: {pct}", end="", flush=True)

    run_scraper(progress_callback=console_progress, should_stop=lambda: False)
    print("\n✅ ニュース収集完了")
