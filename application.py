import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS
import networkx as nx
from itertools import combinations
from collections import Counter

from konlpy.tag import Okt
import koreanize_matplotlib
from matplotlib import font_manager


st.set_page_config(
    page_title="K-POP Demon Hunters 팬덤 분석 대시보드",
    page_icon="🙈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("2025 Netflix 'K-POP Demon Hunters' 팬덤 형성 요인 분석 대시보드")
st.write("학번/이름: C221084 백진웅")
st.caption("데이터: 네이버 검색 API (News/Blog), 키워드-감성-네트워크 기반 분석")
st.divider()


NEWS_PATH = "kpop_demon_hunters_news.csv"
BLOG_PATH = "kpop_demon_hunters_blog.csv"


@st.cache_data
def load_data(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 날짜가 깨진 행이 있으면 이후 필터에서 문제가 생길 수 있어서 먼저 정리함.
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    df = df.dropna(subset=["pubDate"]).copy()

    # 어떤 데이터에서 온 건지 표시해두면 합쳐서 봐도 구분이 된다
    df["source"] = source
    return df


# 뉴스와 블로그는 글 스타일이 달라서 둘 다 준비해두고 선택해서 볼 수 있게 만듬.
df_news = load_data(NEWS_PATH, "news")
df_blog = load_data(BLOG_PATH, "blog")
df_all = pd.concat([df_news, df_blog], ignore_index=True)


st.sidebar.title("분석 옵션")

# 뉴스만 볼지, 블로그만 볼지, 둘 다 합쳐서 볼지 선택할 수 있게 함.
data_type = st.sidebar.selectbox("데이터 타입", ["news", "blog", "all"], index=0)

if data_type == "news":
    df_raw = df_news.copy()
elif data_type == "blog":
    df_raw = df_blog.copy()
else:
    df_raw = df_all.copy()

if len(df_raw) == 0:
    st.error("데이터가 비어있습니다. CSV 파일 내용을 확인하세요.")
    st.stop()

# 데이터가 실제로 존재하는 날짜 안에서만 고르게 함
min_d = df_raw["pubDate"].dt.date.min()
max_d = df_raw["pubDate"].dt.date.max()


def clamp_date_range(d1: date, d2: date, lo: date, hi: date) -> tuple[date, date]:
    # 날짜 입력이 범위를 벗어나면 자동으로 다시 맞춰줌
    if d1 < lo:
        d1 = lo
    if d1 > hi:
        d1 = hi
    if d2 < lo:
        d2 = lo
    if d2 > hi:
        d2 = hi
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


# 처음에는 전체 범위를 한 번에 볼 수 있게 기본값을 잡음
default_start, default_end = clamp_date_range(min_d, max_d, min_d, max_d)

start_date, end_date = st.sidebar.date_input(
    "분석 기간 선택",
    value=(default_start, default_end),
    min_value=min_d,
    max_value=max_d,
    key=f"date_range_{data_type}",
)

# 제목만 볼지 본문까지 같이 볼지에 따라 결과가 달라질 수 있어서 선택하게 함
use_desc = st.sidebar.checkbox("title + description 사용", value=True)

# 워드클라우드는 단어가 너무 많으면 읽기 어려워서 개수를 조절할 수 있게 함
max_words = st.sidebar.slider("워드클라우드 최대 단어수", 10, 200, 50)

# 관계를 너무 약한 것까지 연결하면 화면이 지저분해지니 최소 기준을 둠
min_edge_weight = st.sidebar.slider("네트워크 동시출현 최소 빈도", 2, 50, 10)

# 네트워크 노드 수가 너무 많으면 보기 어려워서 핵심 키워드만 남기도록 함
top_n_nodes = st.sidebar.slider("네트워크에 포함할 상위 키워드 수", 10, 100, 40)

# 감성 점수는 간단한 방식과 주제 중심 방식이 다르게 보일 수 있어서 선택하게 함
sentiment_mode = st.sidebar.radio("감성 기준 선택", ["간단 규칙 기반", "키워드 기반"], index=0)


df = df_raw.copy()

# 사용자가 고른 기간으로 범위를 줄여야 원하는 구간만 분석할 수 있다
df = df[(df["pubDate"].dt.date >= start_date) & (df["pubDate"].dt.date <= end_date)].copy()

# 텍스트를 한 칸에 모아두면 뒤에서 같은 방식으로 처리할 수 있다
if use_desc:
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
else:
    df["text"] = df["title"].fillna("").astype(str)

st.write(f"필터링 후 문서 수: {len(df):,}개")
st.divider()


okt = Okt()


def clean_text_ko(text: str) -> str:
    # 쓸데없는 기호나 태그는 제거
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^가-힣0-9A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_nouns(text: str, stopwords: set) -> list[str]:
    # 핵심 단어만 뽑아야 주제 흐름이 보이기에 명사 위주로 가져옴
    text = clean_text_ko(text)
    nouns = okt.nouns(text)
    nouns = [w for w in nouns if (len(w) > 1) and (w not in stopwords)]
    return nouns


# 검색어 자체는 핵심 요인을 가리는 경우가 많아서 제외함
base_stop = set(["데몬", "헌터스", "넷플릭스", "k팝", "케이팝", "관련", "이번", "통해", "기자"])
stopwords = set(base_stop)


all_nouns = []
for t in df["text"].tolist():
    all_nouns.append(extract_nouns(t, stopwords))


st.subheader("언급량 트렌드")

if len(df) == 0:
    st.warning("현재 설정에서 데이터가 없습니다. 기간이나 데이터 타입을 바꿔보세요.")
else:
    # 날짜별로 묶으면 관심이 언제 집중됐는지 한눈에 볼 수 있음
    trend = df.copy()
    trend["date"] = trend["pubDate"].dt.date
    trend_cnt = trend.groupby("date").size().reset_index(name="count")

    chart = (
        alt.Chart(trend_cnt)
        .mark_line()
        .encode(x="date:T", y="count:Q", tooltip=["date:T", "count:Q"])
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()


# 글의 분위기를 대략 보기 위한 단어 목록을 미리 정해봄
pos_words = set(["흥행", "호평", "인기", "열풍", "감동", "완성도", "기대", "대박", "추천", "화제"])
neg_words = set(["논란", "혹평", "부진", "실망", "비판", "문제", "아쉽", "불만"])

# 팬덤 형성 요인을 설명할 때 자주 나오는 주제 단어를 따로 묶어봄
theme_pos = set(["음악", "노래", "퍼포먼스", "안무", "보컬", "스토리", "캐릭터", "비주얼", "연출", "세계관"])
theme_neg = set(["표절", "논란", "불호", "혹평"])


def sentiment_score_rule(nouns: list[str]) -> int:
    # 긍정 단어는 더하고 부정 단어는 빼서 분위기를 점수로 만듬
    score = 0
    for w in nouns:
        if w in pos_words:
            score += 1
        if w in neg_words:
            score -= 1
    return score


def sentiment_score_keyword(nouns: list[str]) -> int:
    # 주제 단어 중심으로 점수를 만들면 무엇 때문에 반응이 나왔는지 보기 쉬움
    score = 0
    for w in nouns:
        if w in theme_pos:
            score += 1
        if w in theme_neg:
            score -= 1
    return score


st.subheader("감성 분포")

if len(df) == 0:
    st.warning("현재 설정에서 데이터가 없습니다. 기간이나 데이터 타입을 바꿔보세요.")
else:
    # 같은 데이터라도 점수 기준이 달라지면 결과가 달라질 수 있어서 선택 값을 반영함
    if sentiment_mode == "간단 규칙 기반":
        df["sentiment_score"] = [sentiment_score_rule(n) for n in all_nouns]
    else:
        df["sentiment_score"] = [sentiment_score_keyword(n) for n in all_nouns]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["sentiment_score"], bins=15, kde=True, ax=ax)
    ax.set_title("Sentiment Score Distribution")
    st.pyplot(fig)

st.divider()


st.subheader("핵심 키워드")

# 전체 단어를 한 리스트로 모아야 많이 나온 단어를 쉽게 뽑을 수 있음
flat = [w for doc in all_nouns for w in doc]
freq = Counter(flat)

if len(flat) == 0:
    st.warning("키워드가 없습니다. 기간이나 텍스트 범위를 바꿔보세요.")
else:
    # 많이 나온 단어부터 보여주면 사람들이 뭘 중심으로 얘기하는지 빠르게 잡힘
    top_df = pd.DataFrame(freq.most_common(20), columns=["keyword", "count"])
    fig_bar = px.bar(top_df, x="keyword", y="count", title="Top Keywords")
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("워드클라우드")

if len(flat) == 0:
    st.warning("워드클라우드를 만들 수 없습니다.")
else:
    # 한글이 깨지지 않게 폰트를 먼저 잡아둠
    try:
        han_font_path = font_manager.findfont("AppleGothic")
    except:
        han_font_path = None

    
    wc = WordCloud(
        font_path=han_font_path,
        max_words=max_words,
        stopwords=STOPWORDS,
        width=900,
        height=500,
        background_color="white",
    ).generate(" ".join(flat))

    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

st.divider()


st.subheader("키워드 관계망")

if len(flat) == 0:
    st.warning("네트워크를 만들 수 없습니다.")
else:
    # 핵심 키워드만 남기면 관계망이 너무 복잡해지지 않음
    top_keywords = set([w for w, _ in freq.most_common(top_n_nodes)])

    # 같은 글에서 같이 나온 단어는 같은 흐름에서 언급됐을 가능성이 높음
    edge_list = []
    for nouns in all_nouns:
        nouns = list(set(nouns) & top_keywords)
        if len(nouns) > 1:
            edge_list.extend(combinations(sorted(nouns), 2))

    edge_counts = Counter(edge_list)

    # 너무 드문 연결은 우연일 수 있어서 일정 기준 이상만 남김
    filtered_edges = {e: w for e, w in edge_counts.items() if w >= min_edge_weight}

    G = nx.Graph()
    G.add_weighted_edges_from([(u, v, w) for (u, v), w in filtered_edges.items()])

    st.write(f"노드 수: {G.number_of_nodes():,}개")
    st.write(f"엣지 수: {G.number_of_edges():,}개")

    if G.number_of_nodes() == 0:
        st.warning("네트워크가 비어있습니다. 최소 빈도를 낮추거나 상위 키워드 수를 늘려보세요.")
    else:
        # 연결된 단어끼리는 가까이 보이게 배치하면 흐름이 눈에 들어옴
        pos = nx.spring_layout(G, k=0.35, iterations=80, seed=42)
        node_sizes = [G.degree(n) * 120 for n in G.nodes()]
        edge_widths = [G[u][v]["weight"] * 0.06 for u, v in G.edges()]

        fig_net = plt.figure(figsize=(12, 12))
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_size=node_sizes,
            width=edge_widths,
            font_family=plt.rcParams["font.family"],
            font_size=10,
            node_color="skyblue",
            edge_color="gray",
            alpha=0.85,
        )
        plt.title("Keyword Co-occurrence Network")
        plt.axis("off")
        st.pyplot(fig_net)