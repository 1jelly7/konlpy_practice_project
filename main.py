import fitz
import re
from collections import Counter

import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from konlpy.tag import Okt

import pandas as pd

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def clean_text(text):
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def analyze_korean_nouns(text, extra_stopwords=None):
    okt = Okt()
    nouns = okt.nouns(text)
    nouns = [w for w in nouns if len(w) > 1]

    stopwords = set(STOPWORDS)
    stopwords.update([
        "있다", "하다", "되다", "수", "것", "등", "및", "더", "그", "저", "이"
    ])
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    nouns = [w for w in nouns if w not in stopwords]
    return nouns

def nouns_to_frequency_tensor(nouns):
    counter = Counter(nouns)
    words = list(counter.keys())
    freqs = list(counter.values())

    words_tensor = torch.tensor([len(w) for w in words], dtype=torch.int64)
    freqs_tensor = torch.tensor(freqs, dtype=torch.int64)

    return counter, words, freqs, words_tensor, freqs_tensor

def make_wordcloud_from_freq(counter, font_path, output_image="wordcloud_ko.png"):
    wc = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        colormap="viridis",
        collocations=False
    ).generate_from_frequencies(counter)

    wc.to_file(output_image)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return output_image

pdf_path = "sample.pdf"
font_path = r"C:\Windows\Fonts\malgun.ttf"

raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)
nouns = analyze_korean_nouns(cleaned_text, extra_stopwords={"문서", "페이지", "내용"})

counter, words, freqs, words_tensor, freqs_tensor = nouns_to_frequency_tensor(nouns)
make_wordcloud_from_freq(counter, font_path)

print("상위 20개 명사:", counter.most_common(20))
print("단어 길이 Tensor:", words_tensor)
print("빈도 Tensor:", freqs_tensor)

def make_report_tables(counter, top_n=20):
    df = pd.DataFrame(counter.most_common(top_n), columns=["word", "frequency"])
    df["ratio"] = df["frequency"] / df["frequency"].sum()
    return df

def plot_top_words(freq_df, output_image="top_words_bar.png"):
    plt.figure(figsize=(10, 6))
    plt.barh(freq_df["word"][::-1], freq_df["frequency"][::-1], color="steelblue")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.show()
    return output_image

def save_report_artifacts(counter, top_n=20, csv_path="word_frequency_report.csv", bar_path="top_words_bar.png"):
    report_df = make_report_tables(counter, top_n=top_n)
    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_top_words(report_df, bar_path)
    return report_df, csv_path, bar_path

# counter는 이전 단계에서 생성된 Counter 객체 사용
report_df, csv_path, bar_path = save_report_artifacts(counter, top_n=20)

print(report_df)

def save_wordcloud_from_counter(counter, font_path, output_image="wordcloud_report.png"):
    wc = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        collocations=False,
        colormap="viridis"
    ).generate_from_frequencies(counter)

    wc.to_file(output_image)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return output_image

wordcloud_path = save_wordcloud_from_counter(counter, font_path)

top5 = report_df.head(5)

summary_text = (
    f"상위 5개 명사는 {', '.join(top5['word'])}이며, "
    f"가장 빈도가 높은 단어는 '{top5.iloc[0]['word']}'로 {top5.iloc[0]['frequency']}회 등장했다."
)

print(summary_text)