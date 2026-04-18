from __future__ import annotations

import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import jieba
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from matplotlib import font_manager
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

import text_analysis as ta
import llm_viewpoint as vp

load_dotenv()

try:
    from snownlp import SnowNLP
except ImportError:  # pragma: no cover - requirements include snownlp
    SnowNLP = None


API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
VIDEO_API_URL = "https://www.googleapis.com/youtube/v3/videos"
DATA_DIR = Path("data")
CURRENT_COMMENTS_PATH = DATA_DIR / "current_comments.csv"
CURRENT_VIDEO_META_PATH = DATA_DIR / "current_video_meta.json"
CURRENT_VIEWPOINT_STRATEGY_PATH = DATA_DIR / "current_viewpoint_strategy.json"
CURRENT_LLM_ANALYSIS_PATH = DATA_DIR / "current_llm_analysis.json"
CURRENT_ENGLISH_VIEWPOINT_PATH = DATA_DIR / "current_llm_english_viewpoints.json"

STOPWORDS = {
    "the", "and", "you", "that", "this", "with", "for", "are", "was", "but",
    "not", "have", "your", "just", "like", "from", "they", "will", "what", "when",
    "about", "there", "their", "would", "very", "can", "all", "been", "did", "get",
    "got", "has", "had", "how", "its", "let", "may", "more", "most", "much",
    "must", "now", "one", "only", "our", "out", "own", "say", "she", "some",
    "than", "them", "then", "these", "those", "too", "want", "way", "well", "were",
    "who", "why", "also", "into", "could", "should", "other", "after", "before",
    "each", "few", "here", "being", "does", "done", "still", "take", "make", "many",
    "over", "such", "through", "under", "while", "where", "which", "both", "any",
    "down", "even", "first", "going", "gone", "good", "great", "know", "look", "made",
    "need", "really", "right", "same", "since", "thing", "things", "think", "time",
    "using", "went", "work", "yeah", "yes", "yet", "people", "come", "back", "give",
    "day", "lot", "off", "put", "see", "tell", "try", "use", "used", "keep",
    "never", "new", "nothing", "old", "part", "said", "show", "something", "thanks",
    "though", "upon", "wasn", "weren", "won", "wouldn", "couldn", "shouldn",
    "hasn", "haven", "hadn", "doesn", "didn", "isn", "aren", "don", "ain",
    "可以", "一个", "这个", "那个", "我们", "你们", "他们", "还是", "不是", "没有",
    "就是", "真的", "视频", "评论", "感觉", "觉得", "因为", "所以", "如果", "但是",
    "然后", "已经", "什么", "怎么", "这样", "那样", "自己", "现在", "可能", "应该",
    "知道", "还有", "或者", "而且", "虽然", "不过", "其实", "只是", "非常", "比较",
    "一些", "这些", "那些", "的话", "一下", "一样", "一直", "一点", "起来", "出来",
    "过来", "回来", "下去", "上去", "出去", "过去", "那么", "这么", "多么", "如何",
    "怎样", "到底", "究竟", "难道", "居然", "竟然", "果然", "当然", "显然", "确实",
    "的确", "实在", "根本", "完全", "几乎", "大概", "也许", "或许", "似乎", "好像",
    "差不多", "得了", "算了", "好了", "行了", "罢了", "而已", "东西", "时候", "地方",
    "样子", "问题", "办法", "事情", "关系", "道理", "意思", "方面", "情况", "今天",
    "昨天", "明天", "最近", "以前", "以后", "之后", "之前", "刚才", "马上", "终于",
    "永远", "总是", "经常", "偶尔", "从来", "突然", "渐渐", "慢慢", "哈哈", "呵呵",
    "嘿嘿", "嘻嘻", "哎呀", "哎", "啊", "呀", "嘛", "吧", "呢", "哦", "噢", "嗯",
    "哟", "哇", "哎哟", "卧槽", "我去", "靠", "擦", "草", "妈的", "特么", "尼玛",
    "对吧", "好吧", "行吧", "是吧", "呗", "啦", "喽", "咯", "咧", "呦", "嘞",
    "嗬", "嚯", "嗳", "哼", "呸", "切", "嘘", "咦", "喏", "嗯哼", "嗯哪", "嗯呐",
    "然而", "因此", "此外", "另外", "同时", "并且", "于是", "否则", "无论", "不管",
    "尽管", "既然", "即使", "哪怕", "只要", "只有", "除非", "以便", "以免", "以至于",
    "从而", "进而", "反而", "却", "而", "且", "并", "或", "乃", "亦", "则", "故",
    "遂", "苟", "虽", "若", "如", "倘", "使", "令", "让", "把", "被", "给", "对",
    "向", "往", "从", "在", "到", "于", "为", "与", "和", "同", "跟", "比", "按",
    "照", "据", "凭", "沿", "顺", "逆", "朝", "冲", "由", "经", "过", "通过",
    "关于", "至于", "对于", "鉴于", "基于", "出于", "由于", "为了", "用来", "用以",
    "借以", "以此来", "不了", "为啥", "天哪", "我的天", "嗯嗯", "哦哦", "啊啊",
    "么么", "啦啦", "哈哈哈", "哈哈哈哈",
}


st.set_page_config(
    page_title="YouTube 评论舆情分析",
    page_icon="YT",
    layout="wide",
)


def extract_video_id(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if re.fullmatch(r"[\w-]{11}", value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:shorts|embed)/([\w-]{11})", parsed.path)
        if match:
            return match.group(1)
    return value


def fetch_youtube_comments(api_key: str, video_id: str, max_comments: int) -> pd.DataFrame:
    rows: list[dict] = []
    next_page_token = None

    while len(rows) < max_comments:
        params = {
            "key": api_key,
            "videoId": video_id,
            "part": "snippet,replies",
            "textFormat": "plainText",
            "maxResults": min(100, max_comments - len(rows)),
            "order": "relevance",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(API_URL, params=params, timeout=30)
        if response.status_code != 200:
            detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(f"YouTube API 请求失败：{detail}")

        payload = response.json()
        for item in payload.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(
                {
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author": snippet.get("authorDisplayName", ""),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                    "updated_at": snippet.get("updatedAt", ""),
                    "reply_count": item["snippet"].get("totalReplyCount", 0),
                    "is_reply": False,
                }
            )

            for reply in item.get("replies", {}).get("comments", []):
                if len(rows) >= max_comments:
                    break
                reply_snippet = reply["snippet"]
                rows.append(
                    {
                        "comment_id": reply.get("id", ""),
                        "author": reply_snippet.get("authorDisplayName", ""),
                        "text": reply_snippet.get("textDisplay", ""),
                        "like_count": reply_snippet.get("likeCount", 0),
                        "published_at": reply_snippet.get("publishedAt", ""),
                        "updated_at": reply_snippet.get("updatedAt", ""),
                        "reply_count": 0,
                        "is_reply": True,
                    }
                )

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df


def fetch_video_metadata(api_key: str, video_id: str) -> dict:
    params = {"key": api_key, "id": video_id, "part": "snippet"}
    response = requests.get(VIDEO_API_URL, params=params, timeout=30)
    if response.status_code != 200:
        detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"YouTube 视频信息请求失败：{detail}")

    items = response.json().get("items", [])
    if not items:
        return {"video_id": video_id, "title": "", "description": "", "channel_title": "", "tags": []}
    snippet = items[0].get("snippet", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "tags": snippet.get("tags", []),
        "published_at": snippet.get("publishedAt", ""),
    }


def save_video_metadata(metadata: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CURRENT_VIDEO_META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_video_metadata() -> dict:
    if "video_metadata" in st.session_state and st.session_state.video_metadata:
        return dict(st.session_state.video_metadata)
    if CURRENT_VIDEO_META_PATH.exists():
        try:
            return json.loads(CURRENT_VIDEO_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def clean_text(text: str) -> str:
    return ta.clean_text(text)


def tokenize(text: str) -> list[str]:
    return ta.tokenize(text)


def prepared_documents(texts: Iterable[str]) -> list[str]:
    return ta.prepared_documents(texts)


def topic_model(
    docs: list[str],
    model_name: str,
    n_topics: int,
    n_words: int,
    max_features: int,
) -> tuple[pd.DataFrame, object]:
    if model_name == "LDA":
        vectorizer = CountVectorizer(max_features=max_features, min_df=2)
        matrix = vectorizer.fit_transform(docs)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
        matrix = vectorizer.fit_transform(docs)
        if n_topics > min(matrix.shape):
            raise ValueError("NMF 主题数量不能超过有效评论数和特征词数量，请降低主题数量或增加评论数据。")
        model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)

    model.fit(matrix)
    feature_names = vectorizer.get_feature_names_out()
    records = []
    for topic_idx, topic_weights in enumerate(model.components_, start=1):
        top_indices = topic_weights.argsort()[-n_words:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            records.append(
                {
                    "topic": f"Topic {topic_idx}",
                    "rank": rank,
                    "word": feature_names[idx],
                    "weight": float(topic_weights[idx]),
                }
            )
    return pd.DataFrame(records), model


def viewpoint_topic_model(
    docs: list[str],
    source_comments: list[str],
    model_name: str,
    n_topics: int,
    n_words: int,
    max_features: int,
) -> tuple[pd.DataFrame, list[dict]]:
    min_df = 1 if len(docs) < 20 else 2
    if model_name == "LDA":
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        matrix = vectorizer.fit_transform(docs)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
        matrix = vectorizer.fit_transform(docs)
        if n_topics > min(matrix.shape):
            raise ValueError("NMF 主题数量不能超过有效观点评论数和特征词数量，请降低主题数量或增加评论数据。")
        model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)

    model.fit(matrix)
    doc_topic = model.transform(matrix)
    feature_names = vectorizer.get_feature_names_out()
    records = []
    topic_records = []
    for topic_idx, topic_weights in enumerate(model.components_, start=1):
        topic_id = f"T{topic_idx}"
        top_indices = topic_weights.argsort()[-n_words:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        for rank, idx in enumerate(top_indices, start=1):
            records.append(
                {
                    "topic_id": topic_id,
                    "topic": topic_id,
                    "rank": rank,
                    "word": feature_names[idx],
                    "weight": float(topic_weights[idx]),
                }
            )

        doc_indices = doc_topic[:, topic_idx - 1].argsort()[-3:][::-1]
        representative_comments = [source_comments[idx][:180] for idx in doc_indices if idx < len(source_comments)]
        topic_records.append(
            {
                "topic_id": topic_id,
                "top_words": top_words,
                "representative_comments": representative_comments,
            }
        )
    return pd.DataFrame(records), topic_records


def apply_topic_names(topic_df: pd.DataFrame, topic_names: dict[str, dict[str, str]]) -> pd.DataFrame:
    named = topic_df.copy()
    named["topic"] = named["topic_id"].map(lambda topic_id: topic_names.get(topic_id, {}).get("name", topic_id))
    named["summary"] = named["topic_id"].map(lambda topic_id: topic_names.get(topic_id, {}).get("summary", ""))
    return named


def topic_weight_chart(topic_df: pd.DataFrame) -> go.Figure:
    ordered = topic_df.sort_values(["topic", "weight"], ascending=[True, True])
    return px.bar(
        ordered,
        x="weight",
        y="word",
        color="topic",
        facet_col="topic",
        facet_col_wrap=2,
        orientation="h",
        height=max(420, 180 * topic_df["topic"].nunique()),
        title="主题词权重",
    ).update_yaxes(matches=None, showticklabels=True)


def build_semantic_network(docs_tokens: list[list[str]], top_n_words: int, min_edge_weight: int) -> nx.Graph:
    word_counts = Counter(token for tokens in docs_tokens for token in set(tokens))
    vocabulary = {word for word, _ in word_counts.most_common(top_n_words)}
    edge_counts: Counter[tuple[str, str]] = Counter()

    for tokens in docs_tokens:
        unique_tokens = sorted({token for token in tokens if token in vocabulary})
        for idx, source in enumerate(unique_tokens):
            for target in unique_tokens[idx + 1 :]:
                edge_counts[(source, target)] += 1

    graph = nx.Graph()
    for word in vocabulary:
        graph.add_node(word, size=word_counts[word])
    for (source, target), weight in edge_counts.items():
        if weight >= min_edge_weight:
            graph.add_edge(source, target, weight=weight)
    return graph


def network_to_html(graph: nx.Graph) -> str:
    return network_to_plotly_html(graph)


def network_to_plotly_html(graph: nx.Graph) -> str:
    if graph.number_of_nodes() == 0:
        return "<p>没有足够的共现关系生成语义网络。</p>"

    positions = nx.spring_layout(graph, seed=42, k=0.8)
    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#9aa0a6"),
        hoverinfo="none",
        mode="lines",
    )
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    degrees = dict(graph.degree())
    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>词频: {attrs.get('size', 0)}<br>度: {degrees.get(node, 0)}")
        node_size.append(12 + min(attrs.get("size", 1), 80))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size, color="#2f8f83", line=dict(width=1, color="#ffffff")),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=680,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def viewpoint_topic_network_to_html(graph_data: dict) -> str:
    graph = nx.Graph()
    for node in graph_data.get("nodes", []):
        graph.add_node(
            node.get("id"),
            label=node.get("label", node.get("id")),
            group=node.get("group", "观点"),
            size=node.get("size", 1),
            title=node.get("title", ""),
        )
    for edge in graph_data.get("edges", []):
        graph.add_edge(edge.get("source"), edge.get("target"), weight=edge.get("weight", 1))
    if graph.number_of_nodes() == 0:
        return "<p>暂无观点主题网络数据。请先运行观点主题聚类。</p>"

    positions = nx.spring_layout(graph, seed=42, k=0.9)
    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    color_map = {"观点主题": "#2f8f83", "观点维度": "#f59e0b", "观点词": "#2563eb"}
    degrees = dict(graph.degree())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color="rgba(80,80,80,0.35)"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in graph.nodes()],
            y=[positions[node][1] for node in graph.nodes()],
            mode="markers+text",
            text=[graph.nodes[node].get("label", node) for node in graph.nodes()],
            textposition="top center",
            hovertext=[
                f"{graph.nodes[node].get('label', node)}<br>{graph.nodes[node].get('title', '')}<br>关联数: {degrees.get(node, 0)}"
                for node in graph.nodes()
            ],
            hoverinfo="text",
            marker=dict(
                size=[16 + min(int(graph.nodes[node].get("size", 1)), 32) for node in graph.nodes()],
                color=[color_map.get(graph.nodes[node].get("group", "观点"), "#60a5fa") for node in graph.nodes()],
                line=dict(width=1, color="#ffffff"),
            ),
        )
    )
    fig.update_layout(
        height=620,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def sentiment_score(text: str) -> float:
    return ta.sentiment_score(text)


def sentiment_label(score: float) -> str:
    return ta.sentiment_label(score)


def sentiment_polarity(score: float) -> str:
    return ta.sentiment_polarity(score)


def find_font() -> str | None:
    return ta.find_font("zh")


def make_wordcloud(tokens: list[str], language: str = "zh") -> plt.Figure:
    return ta.make_wordcloud(tokens, language)  # type: ignore[arg-type]


def bilingual_topic_chart(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无双语主题数据", showarrow=False)
        return fig

    chart_df = stats.sort_values("合并频次", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        x=chart_df["中文频次"],
        y=chart_df["主题词"],
        name="中文词频",
        orientation="h",
        marker_color="#2f8f83",
    )
    fig.add_bar(
        x=chart_df["英文频次"],
        y=chart_df["主题词"],
        name="英文映射词频",
        orientation="h",
        marker_color="#f59e0b",
    )
    fig.update_layout(
        barmode="stack",
        height=max(420, 26 * len(chart_df)),
        title="双语合并词频",
        xaxis_title="合并频次",
        yaxis_title="核心主题",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    export_df = df.copy()
    for column in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[column]):
            if getattr(export_df[column].dt, "tz", None) is not None:
                export_df[column] = export_df[column].dt.tz_convert(None)
            else:
                export_df[column] = export_df[column].dt.tz_localize(None)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="comments")
    return output.getvalue()


def save_current_comments(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CURRENT_COMMENTS_PATH, index=False, encoding="utf-8-sig")


def render_summary(df: pd.DataFrame) -> None:
    st.subheader("汇总")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("评论数", f"{len(df):,}")
    col2.metric("作者数", f"{df['author'].nunique():,}")
    col3.metric("总点赞", f"{int(df['like_count'].sum()):,}")
    col4.metric("回复数", f"{int(df['is_reply'].sum()):,}")

    if df["published_at"].notna().any():
        first = df["published_at"].min().strftime("%Y-%m-%d")
        latest = df["published_at"].max().strftime("%Y-%m-%d")
        st.caption(f"评论时间范围：{first} 至 {latest}")

    daily = df.dropna(subset=["published_at"]).copy()
    if not daily.empty:
        daily["date"] = daily["published_at"].dt.date
        daily_counts = daily.groupby("date", as_index=False).size()
        fig = px.line(daily_counts, x="date", y="size", markers=True, title="每日评论数量")
        fig.update_layout(xaxis_title="日期", yaxis_title="评论数")
        st.plotly_chart(fig, use_container_width=True)

    top_authors = df.groupby("author", as_index=False).agg(comments=("text", "count"), likes=("like_count", "sum"))
    top_authors = top_authors.sort_values(["comments", "likes"], ascending=False).head(10)
    st.dataframe(top_authors, use_container_width=True, hide_index=True)


def flatten_token_column(df: pd.DataFrame, column: str) -> list[str]:
    tokens: list[str] = []
    if column not in df.columns:
        return tokens
    for value in df[column]:
        if isinstance(value, list):
            tokens.extend(str(item).strip() for item in value if str(item).strip())
        else:
            tokens.extend(str(value or "").split())
    return tokens


def english_alignment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "en_viewpoint_alignments" not in df.columns:
        return pd.DataFrame(columns=["评论", "中文观点词", "英文对应词", "观点维度", "极性", "置信度"])
    for _, row in df.iterrows():
        comment = str(row.get("text", ""))[:140]
        alignments = row.get("en_viewpoint_alignments", [])
        if not isinstance(alignments, list):
            continue
        for item in alignments:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "评论": comment,
                    "中文观点词": item.get("zh", ""),
                    "英文对应词": item.get("en", ""),
                    "观点维度": item.get("dimension", ""),
                    "极性": item.get("polarity", ""),
                    "置信度": item.get("confidence", 0),
                }
            )
    return pd.DataFrame(rows)


def ensure_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sentiment_score", "sentiment_category", "sentiment_polarity", "language"}
    if required.issubset(df.columns):
        return df
    return ta.enrich_dataframe(df)


def render_sentiment_section(df: pd.DataFrame, key_prefix: str) -> None:
    if ta.SnowNLP is None:
        st.warning("当前环境未安装 SnowNLP。运行 `python3 -m pip install -r requirements.txt` 后可获得中文情感评分。")
    if ta.SentimentIntensityAnalyzer is None:
        st.warning("当前环境未安装 vaderSentiment。运行 `python3 -m pip install -r requirements.txt` 后可获得英文情感评分。")

    result_key = f"{key_prefix}_sentiment_df"
    if st.button("运行情感分析", key=f"{key_prefix}_run_sentiment", use_container_width=True):
        st.session_state[result_key] = ta.enrich_dataframe(df.copy())

    if result_key not in st.session_state:
        st.caption("点击按钮后生成情感类别饼图、情感极性饼图和明细表。")
        return

    scored = st.session_state[result_key]
    if scored.empty:
        st.warning("当前数据为空，无法展示情感分析。")
        return

    col_a, col_b, col_c = st.columns(3)
    category_counts = scored["sentiment_category"].value_counts().reset_index()
    category_counts.columns = ["类别", "数量"]
    polarity_counts = scored["sentiment_polarity"].value_counts().reset_index()
    polarity_counts.columns = ["极性", "数量"]
    language_counts = scored["language"].value_counts().reset_index()
    language_counts.columns = ["语言", "数量"]
    col_a.plotly_chart(px.pie(category_counts, names="类别", values="数量", title="情感类别分布"), use_container_width=True)
    col_b.plotly_chart(px.pie(polarity_counts, names="极性", values="数量", title="情感极性分布"), use_container_width=True)
    col_c.plotly_chart(px.pie(language_counts, names="语言", values="数量", title="语言分布"), use_container_width=True)
    st.caption("情感结果统一为正面 / 中性 / 负面；语言列用于区分中文、英文、混合和其他评论。")
    st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button(
        "下载情感分析表格 CSV",
        scored.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key_prefix}_comments_with_sentiment.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_sentiment",
        use_container_width=True,
    )


def render_wordcloud_panels(zh_tokens: list[str], en_tokens: list[str], key_prefix: str) -> None:
    zh_col, en_col = st.columns(2)
    with zh_col:
        st.markdown("**中文词云**")
        zh_top_words = pd.DataFrame(Counter(zh_tokens).most_common(50), columns=["词语", "频次"])
        st.dataframe(zh_top_words, use_container_width=True, hide_index=True)
        if not zh_tokens:
            st.caption("暂无中文词频数据。")
        elif not ta.wordcloud_font_ready("zh"):
            st.warning("未找到可用中文字体，无法安全生成中文词云。")
        else:
            st.pyplot(make_wordcloud(zh_tokens, "zh"), use_container_width=True)
    with en_col:
        st.markdown("**英文词云**")
        en_top_words = pd.DataFrame(Counter(en_tokens).most_common(50), columns=["词语", "频次"])
        st.dataframe(en_top_words, use_container_width=True, hide_index=True)
        if not en_tokens:
            st.caption("暂无英文词频数据。")
        else:
            st.pyplot(make_wordcloud(en_tokens, "en"), use_container_width=True)


def render_bilingual_merge(zh_tokens: list[str], en_tokens: list[str], key_prefix: str, top_n: int = 50) -> None:
    st.caption(
        "英文词先经过 NLTK 停用词、补充停用词和词形还原，再映射或免费翻译成中文主题，与中文核心词合并统计。"
    )
    if not zh_tokens or not en_tokens:
        st.caption("当前数据缺少中文或英文核心词，无法生成双语合并结果。")
        return
    if not ta.wordcloud_font_ready("zh"):
        st.warning("未找到可用中文字体，无法安全生成双语词云。")
        return

    bilingual_stats = ta.build_bilingual_topic_stats(zh_tokens, en_tokens, top_n=top_n)
    st.session_state[f"{key_prefix}_bilingual_stats"] = bilingual_stats
    st.dataframe(bilingual_stats, use_container_width=True, hide_index=True)
    st.plotly_chart(bilingual_topic_chart(bilingual_stats.head(30)), use_container_width=True)
    frequencies = ta.bilingual_wordcloud_frequencies(bilingual_stats)
    if frequencies:
        st.pyplot(ta.make_wordcloud_from_frequencies(frequencies), use_container_width=True)


def render_traditional_topic_tab(
    docs: list[str],
    language: str,
    model_name: str,
    key_prefix: str,
) -> None:
    st.caption(
        f"{model_name} 使用传统分词后的{language}评论建模，适合查看未经过 DeepSeek 观点过滤的基础主题结构。"
    )
    col_a, col_b, col_c = st.columns(3)
    n_topics = col_a.slider("主题数量", 2, 12, 5, key=f"{key_prefix}_topics", help="建议 3~8")
    n_words = col_b.slider("每个主题词数", 5, 25, 12, key=f"{key_prefix}_words")
    max_features = col_c.slider("最大特征词", 100, 5000, 1200, step=100, key=f"{key_prefix}_features")
    result_key = f"{key_prefix}_topic_df"

    if st.button(f"运行传统 {model_name} 主题模型", key=f"{key_prefix}_run", use_container_width=True):
        topic_docs = [doc for doc in docs if doc.strip()]
        if len(topic_docs) < 2:
            st.warning(f"{language}有效评论太少，无法运行主题模型。")
        else:
            try:
                topic_df, _ = topic_model(topic_docs, model_name, n_topics, n_words, max_features)
                st.session_state[result_key] = topic_df
                st.session_state[f"{result_key}_language"] = language
            except ValueError as exc:
                st.error(f"主题模型无法运行：{exc}")

    if result_key in st.session_state:
        st.caption(f"当前主题模型语言：{st.session_state.get(f'{result_key}_language', language)}")
        st.dataframe(st.session_state[result_key], use_container_width=True, hide_index=True)
        st.plotly_chart(topic_weight_chart(st.session_state[result_key]), use_container_width=True)
        st.download_button(
            "下载主题词权重 CSV",
            st.session_state[result_key].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{key_prefix}_topic_words.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
            use_container_width=True,
        )


def render_llm_topic_tab(
    viewpoint_df: pd.DataFrame,
    video_metadata: dict,
    strategy: dict,
    deepseek_api_key: str,
    deepseek_model: str,
    view_language: str,
    model_name: str,
    key_prefix: str,
) -> None:
    st.caption(
        f"{model_name} 只使用 DeepSeek 动态策略过滤后的观点文本建模，主题名称由 DeepSeek 根据视频语境自动命名。"
    )
    col_a, col_b, col_c = st.columns(3)
    n_topics = col_a.slider("观点主题数", 2, 10, 5, key=f"{key_prefix}_topics")
    n_words = col_b.slider("每主题观点词", 5, 20, 10, key=f"{key_prefix}_words")
    max_features = col_c.slider("观点特征词", 50, 2000, 600, step=50, key=f"{key_prefix}_features")
    result_key = f"{key_prefix}_topic_df"

    if st.button(f"运行 DeepSeek 观点 {model_name}", key=f"{key_prefix}_run", use_container_width=True):
        filtered_docs, source_comments, _, _ = vp.purified_viewpoint_docs_and_tokens(
            viewpoint_df,
            view_language,
            strategy,
        )
        if len(filtered_docs) < 2:
            st.warning("有效观点评论太少，无法运行观点主题模型。")
        else:
            try:
                topic_df, topic_records = viewpoint_topic_model(
                    filtered_docs,
                    source_comments,
                    model_name,
                    n_topics,
                    n_words,
                    max_features,
                )
                topic_names = vp.name_topics_with_deepseek(
                    topic_records,
                    video_metadata,
                    strategy,
                    deepseek_api_key,
                    model=deepseek_model,
                )
                named_topic_df = apply_topic_names(topic_df, topic_names)
                analysis = vp.analysis_to_jsonable(
                    strategy,
                    viewpoint_df,
                    named_topic_df,
                    topic_names,
                    topic_records,
                    view_language,
                )
                analysis["model"] = model_name
                analysis["generated_at"] = datetime.now().isoformat(timespec="seconds")
                cache_key = vp.build_analysis_cache_key(video_metadata, strategy, viewpoint_df)
                vp.write_cached_analysis(CURRENT_LLM_ANALYSIS_PATH, cache_key, analysis)
                st.session_state[result_key] = named_topic_df
                st.session_state[f"{key_prefix}_topic_records"] = topic_records
                st.session_state[f"{key_prefix}_topic_names"] = topic_names
                st.session_state[f"{key_prefix}_analysis"] = analysis
                st.session_state.llm_latest_analysis = analysis
                st.success("已生成观点主题模型，并写入 DeepSeek 分析缓存。")
            except ValueError as exc:
                st.error(f"观点主题模型无法运行：{exc}")
            except Exception as exc:
                st.error(f"观点主题聚类失败：{exc}")

    if result_key in st.session_state:
        st.dataframe(st.session_state[result_key], use_container_width=True, hide_index=True)
        st.plotly_chart(topic_weight_chart(st.session_state[result_key]), use_container_width=True)
        summaries = pd.DataFrame(
            [
                {"观点主题": value.get("name", key), "解释": value.get("summary", "")}
                for key, value in st.session_state.get(f"{key_prefix}_topic_names", {}).items()
            ]
        )
        if not summaries.empty:
            st.dataframe(summaries, use_container_width=True, hide_index=True)


def load_llm_analysis(video_metadata: dict, strategy: dict, viewpoint_df: pd.DataFrame) -> dict | None:
    if "llm_latest_analysis" in st.session_state and st.session_state.llm_latest_analysis:
        return st.session_state.llm_latest_analysis
    cache_key = vp.build_analysis_cache_key(video_metadata, strategy, viewpoint_df)
    return vp.read_cached_analysis(CURRENT_LLM_ANALYSIS_PATH, cache_key)


def main() -> None:
    st.title("YouTube 评论舆情分析")
    st.caption("抓取评论，生成表格，并完成主题模型、语义网络、情感和词云分析。")

    existing_metadata = load_video_metadata()
    with st.sidebar:
        st.header("数据抓取")
        youtube_api_key_env = os.getenv("YOUTUBE_API_KEY", "").strip()
        api_key_input = st.text_input("YouTube Data API Key（可留空读取 .env）", type="password")
        api_key = api_key_input.strip() or youtube_api_key_env
        if youtube_api_key_env and not api_key_input.strip():
            st.caption("已从 .env / 环境变量读取 YouTube Data API Key。")
        video_value = st.text_input("视频链接或视频 ID")
        max_comments = st.number_input("最大评论数", min_value=10, max_value=5000, value=300, step=50)
        fetch_clicked = st.button("抓取评论", type="primary", use_container_width=True)

        st.divider()
        uploaded_file = st.file_uploader("或上传已有评论表格", type=["csv", "xlsx"])

        st.divider()
        st.header("DeepSeek 观点分析")
        deepseek_key_input = st.text_input("DeepSeek API Key（可留空读取 .env）", type="password")
        deepseek_api_key = deepseek_key_input.strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
        if os.getenv("DEEPSEEK_API_KEY", "").strip() and not deepseek_key_input.strip():
            st.caption("已从 .env / 环境变量读取 DeepSeek API Key。")
        deepseek_model_options = ["deepseek-chat", "deepseek-reasoner"]
        deepseek_model_default = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        deepseek_model_index = deepseek_model_options.index(deepseek_model_default) if deepseek_model_default in deepseek_model_options else 0
        deepseek_model = st.selectbox("DeepSeek 模型", deepseek_model_options, index=deepseek_model_index)
        force_viewpoint_refresh = st.checkbox("重新生成观点策略", value=False)

        st.divider()
        st.header("视频上下文")
        if existing_metadata.get("title") or existing_metadata.get("description"):
            st.caption("抓取视频时会自动从 YouTube Data API 获取标题和描述。")
            st.caption(f"当前标题：{str(existing_metadata.get('title', ''))[:80]}")
        manual_context_enabled = st.checkbox(
            "上传表格时手动补充/覆盖视频上下文",
            value=not bool(existing_metadata.get("title") or existing_metadata.get("description")),
        )
        manual_title = str(existing_metadata.get("title", ""))
        manual_description = str(existing_metadata.get("description", ""))
        manual_channel = str(existing_metadata.get("channel_title", ""))
        if manual_context_enabled:
            manual_title = st.text_input("视频标题", value=manual_title)
            manual_description = st.text_area("视频描述", value=manual_description, height=120)
            manual_channel = st.text_input("频道名称", value=manual_channel)

    if "comments_df" not in st.session_state:
        st.session_state.comments_df = pd.DataFrame()

    if fetch_clicked:
        video_id = extract_video_id(video_value)
        if not api_key or not video_id:
            st.error("请填写 API Key 和视频链接或 ID。")
        else:
            with st.spinner("正在通过 YouTube Data API 抓取评论..."):
                try:
                    st.session_state.comments_df = fetch_youtube_comments(api_key, video_id, int(max_comments))
                    try:
                        st.session_state.video_metadata = fetch_video_metadata(api_key, video_id)
                        save_video_metadata(st.session_state.video_metadata)
                    except Exception as meta_exc:
                        st.warning(f"评论已抓取，但视频标题/描述获取失败：{meta_exc}")
                    save_current_comments(st.session_state.comments_df)
                    st.success(f"已抓取 {len(st.session_state.comments_df)} 条评论。")
                except Exception as exc:
                    st.error(str(exc))

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.comments_df = pd.read_csv(uploaded_file)
        else:
            st.session_state.comments_df = pd.read_excel(uploaded_file)
        if "published_at" in st.session_state.comments_df.columns:
            st.session_state.comments_df["published_at"] = pd.to_datetime(
                st.session_state.comments_df["published_at"], errors="coerce"
            )
        save_current_comments(st.session_state.comments_df)

    video_metadata = load_video_metadata()
    if manual_context_enabled and (manual_title.strip() or manual_description.strip() or manual_channel.strip()):
        video_metadata.update(
            {
                "title": manual_title.strip(),
                "description": manual_description.strip(),
                "channel_title": manual_channel.strip(),
                "video_id": video_metadata.get("video_id", extract_video_id(video_value)),
            }
        )
        st.session_state.video_metadata = video_metadata
        save_video_metadata(video_metadata)

    df = st.session_state.comments_df
    if df.empty:
        st.info("先在侧边栏抓取评论，或上传包含 text 列的评论表格。")
        return

    if "text" not in df.columns:
        st.error("表格中需要包含 text 列。")
        return

    st.subheader("评论表格")
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.download_button(
        "下载 CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"youtube_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "下载 Excel",
        to_excel_bytes(df),
        file_name=f"youtube_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    render_summary(df)

    texts = df["text"].fillna("").astype(str).tolist()
    zh_docs = ta.prepared_documents(texts, "zh")
    en_docs = ta.prepared_documents(texts, "en")
    zh_token_lists = [doc.split() for doc in zh_docs if doc.strip()]
    en_token_lists = [doc.split() for doc in en_docs if doc.strip()]
    zh_tokens = [token for tokens in zh_token_lists for token in tokens]
    en_tokens = [token for tokens in en_token_lists for token in tokens]
    all_tokens = zh_tokens + en_tokens

    st.subheader("分析")
    if len(all_tokens) < 10:
        st.warning("可分析词语太少，请增加评论数量或检查文本列。")
        return

    analysis_module = st.radio(
        "分析模块",
        ["DeepSeek 大模型分析", "传统分析"],
        horizontal=True,
        key="analysis_module_choice",
    )

    if analysis_module == "DeepSeek 大模型分析":
        st.info(
            "大模型模块先根据视频标题、描述、标签和评论样本生成本视频专属清洗策略，"
            "再只把观点、评价、情感、体验和建议类表达送入 LDA/NMF、观点网络、词云和双语合并分析。"
        )
        if not deepseek_api_key and "viewpoint_strategy" not in st.session_state:
            st.warning("未配置 DeepSeek API Key。已有匹配缓存时可直接加载；没有缓存时需要在侧边栏配置 Key。")

        meta_col, action_col = st.columns([2, 1])
        current_strategy = st.session_state.get("viewpoint_strategy", {})
        with meta_col:
            st.markdown(f"**视频类型**：{current_strategy.get('video_type', '待生成')}")
            st.markdown(f"**核心主题**：{current_strategy.get('core_theme', '待生成')}")
        with action_col:
            generate_viewpoint = st.button("加载 / 生成观点策略", type="primary", use_container_width=True)

        if generate_viewpoint:
            if not deepseek_api_key and (force_viewpoint_refresh or not CURRENT_VIEWPOINT_STRATEGY_PATH.exists()):
                st.error("请先在侧边栏填写 DeepSeek API Key，或取消重新生成后使用已有缓存。")
            else:
                with st.spinner("正在加载或生成视频专属观点清洗策略..."):
                    try:
                        strategy, from_cache = vp.build_viewpoint_strategy(
                            df,
                            video_metadata,
                            deepseek_api_key,
                            model=deepseek_model,
                            cache_path=CURRENT_VIEWPOINT_STRATEGY_PATH,
                            force_refresh=force_viewpoint_refresh,
                        )
                        st.session_state.viewpoint_strategy = strategy
                        st.session_state.viewpoint_df = vp.apply_viewpoint_strategy(
                            df,
                            strategy,
                            video_meta=video_metadata,
                            api_key=deepseek_api_key,
                            model=deepseek_model,
                            english_cache_path=CURRENT_ENGLISH_VIEWPOINT_PATH,
                            use_deepseek_english=True,
                            force_english_refresh=force_viewpoint_refresh,
                        )
                        st.success("已加载缓存策略。" if from_cache else "已生成新的观点策略。")
                    except Exception as exc:
                        st.error(f"观点策略生成失败：{exc}")

        if "viewpoint_strategy" in st.session_state:
            strategy = st.session_state.viewpoint_strategy
            if (
                "viewpoint_df" not in st.session_state
                or len(st.session_state.viewpoint_df) != len(df)
                or not vp.has_viewpoint_columns(st.session_state.viewpoint_df)
            ):
                st.session_state.viewpoint_df = vp.apply_viewpoint_strategy(
                    df,
                    strategy,
                    video_meta=video_metadata,
                    api_key=deepseek_api_key,
                    model=deepseek_model,
                    english_cache_path=CURRENT_ENGLISH_VIEWPOINT_PATH,
                    use_deepseek_english=True,
                    force_english_refresh=False,
                )
            viewpoint_df = st.session_state.viewpoint_df
            dimensions = strategy.get("viewpoint_dimensions", [])
            st.caption("观点维度：" + " / ".join(str(item) for item in dimensions))
            with st.expander("查看 DeepSeek 动态清洗策略", expanded=False):
                st.json(strategy)

            view_language = st.radio("观点分析语言", ["全部", "中文", "英文"], horizontal=True, key="llm_view_language")
            viewpoint_docs, _, viewpoint_token_lists, viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(
                viewpoint_df,
                view_language,
                strategy,
            )
            _, _, _, zh_viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(viewpoint_df, "中文", strategy)
            _, _, _, en_viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(viewpoint_df, "英文", strategy)
            anchor_count = len(flatten_token_column(viewpoint_df, "anchor_tokens"))

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("观点词数量", f"{len(viewpoint_tokens):,}")
            metric_b.metric("观点评论数", f"{sum(bool(doc.strip()) for doc in viewpoint_docs):,}")
            metric_c.metric("锚点词数量", f"{anchor_count:,}")
            zh_mode = (
                viewpoint_df.get("zh_viewpoint_filter_mode", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .mode()
            )
            en_mode = (
                viewpoint_df.get("en_viewpoint_filter_mode", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .mode()
            )
            zh_mode_text = zh_mode.iloc[0] if not zh_mode.empty else "viewpoint_terms"
            en_mode_text = en_mode.iloc[0] if not en_mode.empty else "viewpoint_terms"
            adaptive_modes = {"viewpoint_terms_plus_opinion", "opinion_fallback", "insufficient_viewpoint"}
            if {zh_mode_text, en_mode_text}.intersection(adaptive_modes):
                st.caption(
                    "观点词匹配过少时已自动启用自适应模式：只补充评价、情绪、体验、建议类候选词，"
                    "不会退回原始高频名词。"
                )
            if "en_viewpoint_extraction_mode" in viewpoint_df.columns:
                extraction_counts = viewpoint_df["en_viewpoint_extraction_mode"].value_counts().to_dict()
                if extraction_counts.get("deepseek_aligned", 0):
                    st.caption(
                        f"英文观点抽取：DeepSeek 对齐 {int(extraction_counts.get('deepseek_aligned', 0))} 条，"
                        f"本地回退 {int(extraction_counts.get('local_fallback', 0))} 条。"
                    )

            if not viewpoint_tokens:
                st.warning("当前筛选下观点词过少。系统已启用观点候选回退，但不会退回原始高频名词。")
            else:
                (
                    llm_lda_tab,
                    llm_nmf_tab,
                    llm_network_tab,
                    llm_sentiment_tab,
                    llm_wordcloud_tab,
                    llm_bilingual_tab,
                ) = st.tabs(
                    ["LDA主题模型", "NMF主题模型", "networkx语义网络", "情感分析", "词云图", "双语合并分析"]
                )
                with llm_lda_tab:
                    render_llm_topic_tab(
                        viewpoint_df,
                        video_metadata,
                        strategy,
                        deepseek_api_key,
                        deepseek_model,
                        view_language,
                        "LDA",
                        "llm_lda",
                    )
                with llm_nmf_tab:
                    render_llm_topic_tab(
                        viewpoint_df,
                        video_metadata,
                        strategy,
                        deepseek_api_key,
                        deepseek_model,
                        view_language,
                        "NMF",
                        "llm_nmf",
                    )
                with llm_network_tab:
                    st.caption("这里展示的是二次净化后的观点词共现，以及观点主题与 DeepSeek 观点维度之间的关系。")
                    co_network = vp.build_viewpoint_cooccurrence_network(viewpoint_df, strategy, view_language)
                    if not co_network.get("nodes"):
                        st.info("当前筛选下暂无可用观点共现网络。")
                    else:
                        st.json(
                            {
                                "观点共现节点数": len(co_network.get("nodes", [])),
                                "观点共现边数": len(co_network.get("edges", [])),
                                "语言": view_language,
                            }
                        )
                        components.html(viewpoint_topic_network_to_html(co_network), height=700, scrolling=True)
                        st.download_button(
                            "下载观点共现网络 HTML",
                            viewpoint_topic_network_to_html(co_network).encode("utf-8"),
                            file_name="deepseek_viewpoint_cooccurrence_network.html",
                            mime="text/html",
                            key="llm_download_cooccurrence_network",
                            use_container_width=True,
                        )

                    st.divider()
                    analysis = load_llm_analysis(video_metadata, strategy, viewpoint_df)
                    if not analysis or not analysis.get("topic_network", {}).get("nodes"):
                        st.info("运行 LDA 或 NMF 后，这里还会显示自动命名主题与观点维度网络。")
                    else:
                        network_data = analysis.get("topic_network", {})
                        st.subheader("主题维度网络")
                        st.json(
                            {
                                "节点数": len(network_data.get("nodes", [])),
                                "边数": len(network_data.get("edges", [])),
                                "模型": analysis.get("model", "未知"),
                                "语言": analysis.get("language", "全部"),
                            }
                        )
                        components.html(viewpoint_topic_network_to_html(network_data), height=700, scrolling=True)
                        st.download_button(
                            "下载 DeepSeek 观点网络 HTML",
                            viewpoint_topic_network_to_html(network_data).encode("utf-8"),
                            file_name="deepseek_viewpoint_network.html",
                            mime="text/html",
                            key="llm_download_network",
                            use_container_width=True,
                        )
                with llm_sentiment_tab:
                    st.caption("情感分析仍使用中英文各自模型，但只统计被 DeepSeek 识别出观点表达的评论。")
                    viewpoint_rows = vp.viewpoint_rows_with_purified_tokens(viewpoint_df, view_language, strategy)
                    render_sentiment_section(viewpoint_rows, "llm_viewpoint")
                with llm_wordcloud_tab:
                    st.caption("词云只显示 DeepSeek 动态策略保留下来的观点词。")
                    render_wordcloud_panels(zh_viewpoint_tokens, en_viewpoint_tokens, "llm_viewpoint")
                    alignment_df = english_alignment_dataframe(viewpoint_df)
                    if not alignment_df.empty:
                        with st.expander("查看英文评论的中文观点词与英文原词对齐", expanded=False):
                            st.dataframe(alignment_df, use_container_width=True, hide_index=True)
                with llm_bilingual_tab:
                    st.caption("中文观点词和英文观点词分别清洗，再把英文观点词映射为中文主题做合并展示。")
                    render_bilingual_merge(zh_viewpoint_tokens, en_viewpoint_tokens, "llm_viewpoint", top_n=50)
        else:
            st.caption("点击“加载 / 生成观点策略”后进入 DeepSeek 大模型分析。")

    else:
        st.info(
            "传统模块保留原有中英文分开处理逻辑：中文分词、英文词形还原和停用词过滤分别进行，"
            "并提供 LDA、NMF、networkx 共现网络、情感、词云和双语合并分析。"
        )
        traditional_language = st.radio("传统分析语言", ["中文", "英文"], horizontal=True, key="traditional_language_choice")
        language_code = "zh" if traditional_language == "中文" else "en"
        traditional_docs = zh_docs if traditional_language == "中文" else en_docs
        traditional_token_lists = zh_token_lists if traditional_language == "中文" else en_token_lists

        (
            trad_lda_tab,
            trad_nmf_tab,
            trad_network_tab,
            trad_sentiment_tab,
            trad_wordcloud_tab,
            trad_bilingual_tab,
        ) = st.tabs(["LDA主题模型", "NMF主题模型", "networkx语义网络", "情感分析", "词云图", "双语合并分析"])

        with trad_lda_tab:
            render_traditional_topic_tab(traditional_docs, traditional_language, "LDA", f"traditional_lda_{language_code}")
        with trad_nmf_tab:
            render_traditional_topic_tab(traditional_docs, traditional_language, "NMF", f"traditional_nmf_{language_code}")
        with trad_network_tab:
            st.caption("传统语义网络基于同一评论内的词语共现关系构建，中文和英文分开处理。")
            col_a, col_b = st.columns(2)
            top_n_words = col_a.slider("网络词数量", 20, 200, 80, step=10, key=f"traditional_network_words_{language_code}")
            min_edge_weight = col_b.slider("最小共现次数", 1, 20, 2, key=f"traditional_network_edges_{language_code}")
            network_key = f"traditional_network_{language_code}"
            if st.button("生成语义网络", key=f"{network_key}_run", use_container_width=True):
                graph = build_semantic_network(traditional_token_lists, top_n_words, min_edge_weight)
                st.session_state[f"{network_key}_html"] = network_to_html(graph)
                st.session_state[f"{network_key}_stats"] = {
                    "节点数": graph.number_of_nodes(),
                    "边数": graph.number_of_edges(),
                    "密度": round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0,
                }
            if f"{network_key}_html" in st.session_state:
                st.json(st.session_state[f"{network_key}_stats"])
                components.html(st.session_state[f"{network_key}_html"], height=720, scrolling=True)
                st.download_button(
                    "下载语义网络 HTML",
                    st.session_state[f"{network_key}_html"].encode("utf-8"),
                    file_name=f"traditional_{language_code}_semantic_network.html",
                    mime="text/html",
                    key=f"{network_key}_download",
                    use_container_width=True,
                )
        with trad_sentiment_tab:
            st.caption("自动识别评论语言：中文使用 SnowNLP，英文使用 VADER，统一输出正面 / 中性 / 负面。")
            render_sentiment_section(df.copy(), "traditional")
        with trad_wordcloud_tab:
            st.caption("中文词云和英文词云分别生成，中文词云会自动查找可用中文字体以避免乱码。")
            render_wordcloud_panels(zh_tokens, en_tokens, "traditional")
        with trad_bilingual_tab:
            st.caption("保留中英文各自清洗规则，再将英文核心词映射为中文主题，形成合并词频图和双语词云。")
            render_bilingual_merge(zh_tokens, en_tokens, "traditional", top_n=50)

    st.divider()
    st.link_button("进入数据展示大屏 →", "/dashboard", use_container_width=True)


if __name__ == "__main__":
    main()
