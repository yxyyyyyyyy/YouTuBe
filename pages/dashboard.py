from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

import jieba
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import text_analysis as ta
import llm_viewpoint as vp

try:
    from snownlp import SnowNLP
except ImportError:
    SnowNLP = None


st.set_page_config(page_title="YouTube 评论舆情数据大屏", page_icon="DATA", layout="wide")

CURRENT_COMMENTS_PATH = Path("data/current_comments.csv")
CURRENT_VIDEO_META_PATH = Path("data/current_video_meta.json")
CURRENT_VIEWPOINT_STRATEGY_PATH = Path("data/current_viewpoint_strategy.json")
CURRENT_LLM_ANALYSIS_PATH = Path("data/current_llm_analysis.json")
CURRENT_ENGLISH_VIEWPOINT_PATH = Path("data/current_llm_english_viewpoints.json")

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

COLORS = {
    "bg": "#0f1117",
    "panel": "#1a1d2e",
    "panel_2": "#1e2235",
    "text": "#e8eaed",
    "muted": "#9aa0a6",
    "teal": "#2dd4bf",
    "red": "#f87171",
    "yellow": "#facc15",
    "blue": "#60a5fa",
    "purple": "#a78bfa",
    "green": "#4ade80",
    "border": "rgba(255,255,255,0.08)",
}


def inject_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {COLORS["bg"]} !important;
            color: {COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: #12141f !important;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .stHeader {{
            color: #d4d4d8 !important;
        }}
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stFileUploader label {{
            color: #c4c4c8 !important;
        }}
        .block-container {{
            max-width: 1720px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            background: {COLORS["bg"]} !important;
        }}
        [data-testid="stHeader"] {{
            background: {COLORS["bg"]} !important;
        }}
        [data-testid="stToolbar"] {{
            background: {COLORS["bg"]} !important;
        }}
        .stPlotlyChart {{
            background: {COLORS["panel"]} !important;
            border-radius: 8px;
        }}
        .js-plotly-plot .plotly .modebar {{
            background: transparent !important;
        }}
        .hero {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 20px;
            padding: 20px 0 16px;
            border-bottom: 2px solid rgba(45,212,191,0.3);
            margin-bottom: 20px;
        }}
        .hero h1 {{
            color: #ffffff;
            font-size: 32px;
            line-height: 1.15;
            letter-spacing: 0.5px;
            margin: 0;
        }}
        .hero p {{
            color: #b0b4bc;
            font-size: 14px;
            margin: 8px 0 0;
        }}
        .badge {{
            border: 1px solid rgba(45,212,191,0.55);
            background: rgba(45,212,191,0.12);
            color: #ccfbf1;
            border-radius: 8px;
            padding: 8px 14px;
            white-space: nowrap;
            font-size: 13px;
        }}
        .metric-card {{
            background: {COLORS["panel_2"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
            padding: 18px 16px 14px;
            min-height: 120px;
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            border-radius: 10px 10px 0 0;
        }}
        .metric-card.accent-teal::before {{ background: {COLORS["teal"]}; }}
        .metric-card.accent-yellow::before {{ background: {COLORS["yellow"]}; }}
        .metric-card.accent-blue::before {{ background: {COLORS["blue"]}; }}
        .metric-card.accent-red::before {{ background: {COLORS["red"]}; }}
        .metric-card.accent-purple::before {{ background: {COLORS["purple"]}; }}
        .metric-card .label {{
            color: #b8bcc4;
            font-size: 13px;
            margin-bottom: 10px;
            font-weight: 500;
        }}
        .metric-card .value {{
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            line-height: 1.1;
        }}
        .metric-card .note {{
            color: #8b8f96;
            font-size: 12px;
            margin-top: 10px;
        }}
        .section-block {{
            margin-bottom: 6px;
        }}
        .section-title {{
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            margin: 4px 0 2px;
            padding-left: 12px;
            border-left: 3px solid {COLORS["teal"]};
            line-height: 1.4;
        }}
        .section-desc {{
            color: #8b919a;
            font-size: 12px;
            margin: 2px 0 10px;
            padding-left: 15px;
            line-height: 1.5;
        }}
        .comment-card {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }}
        .comment-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }}
        .comment-author {{
            color: {COLORS["teal"]};
            font-size: 13px;
            font-weight: 700;
        }}
        .comment-likes {{
            color: #8b8f96;
            font-size: 12px;
        }}
        .sentiment-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            line-height: 1.4;
        }}
        .sentiment-tag.positive {{
            background: rgba(45,212,191,0.18);
            color: {COLORS["teal"]};
            border: 1px solid rgba(45,212,191,0.3);
        }}
        .sentiment-tag.neutral {{
            background: rgba(250,204,21,0.15);
            color: {COLORS["yellow"]};
            border: 1px solid rgba(250,204,21,0.3);
        }}
        .sentiment-tag.negative {{
            background: rgba(248,113,113,0.15);
            color: {COLORS["red"]};
            border: 1px solid rgba(248,113,113,0.3);
        }}
        .comment-text {{
            color: #d4d8e0;
            font-size: 13px;
            line-height: 1.55;
            margin-top: 2px;
        }}
        .author-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 4px;
        }}
        .author-table th {{
            color: #9aa0a6;
            font-size: 12px;
            font-weight: 600;
            text-align: left;
            padding: 6px 10px;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        .author-table td {{
            color: #d4d8e0;
            font-size: 13px;
            padding: 8px 10px;
            background: {COLORS["panel"]};
        }}
        .author-table tr td:first-child {{
            border-radius: 6px 0 0 6px;
            color: {COLORS["teal"]};
            font-weight: 600;
        }}
        .author-table tr td:last-child {{
            border-radius: 0 6px 6px 0;
        }}
        .rank-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            font-size: 11px;
            font-weight: 700;
            margin-right: 6px;
        }}
        .rank-badge.gold {{ background: rgba(250,204,21,0.25); color: {COLORS["yellow"]}; }}
        .rank-badge.silver {{ background: rgba(148,163,184,0.25); color: #94a3b8; }}
        .rank-badge.bronze {{ background: rgba(251,146,60,0.25); color: #fb923c; }}
        .rank-badge.normal {{ background: rgba(255,255,255,0.08); color: #71717a; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(text: str) -> str:
    return ta.clean_text(text)


def tokenize(text: str) -> list[str]:
    return ta.tokenize(text)


def sentiment_score(text: str) -> float:
    return ta.sentiment_score(text)


def sentiment_label(score: float) -> str:
    return ta.sentiment_label(score)


def sentiment_polarity(score: float) -> str:
    return ta.sentiment_polarity(score)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "published_at" in normalized.columns:
        normalized["published_at"] = pd.to_datetime(normalized["published_at"], errors="coerce")
    else:
        normalized["published_at"] = pd.NaT
    normalized["like_count"] = pd.to_numeric(normalized.get("like_count", 0), errors="coerce").fillna(0)
    normalized["reply_count"] = pd.to_numeric(normalized.get("reply_count", 0), errors="coerce").fillna(0)
    if "author" not in normalized.columns:
        normalized["author"] = "未知作者"
    if "text" not in normalized.columns:
        normalized["text"] = ""
    return normalized


def get_source_data() -> pd.DataFrame:
    if "comments_df" in st.session_state and not st.session_state.comments_df.empty:
        return normalize_dataframe(st.session_state.comments_df)
    if CURRENT_COMMENTS_PATH.exists():
        return normalize_dataframe(pd.read_csv(CURRENT_COMMENTS_PATH))
    return pd.DataFrame()


def get_viewpoint_strategy() -> dict | None:
    if "viewpoint_strategy" in st.session_state and st.session_state.viewpoint_strategy:
        return vp.normalize_strategy(st.session_state.viewpoint_strategy)
    if CURRENT_VIEWPOINT_STRATEGY_PATH.exists():
        try:
            payload = json.loads(CURRENT_VIEWPOINT_STRATEGY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
        strategy = payload.get("strategy", payload) if isinstance(payload, dict) else None
        if isinstance(strategy, dict):
            return vp.normalize_strategy(strategy)
    return None


def get_video_metadata() -> dict:
    if "video_metadata" in st.session_state and st.session_state.video_metadata:
        return dict(st.session_state.video_metadata)
    if CURRENT_VIDEO_META_PATH.exists():
        try:
            payload = json.loads(CURRENT_VIDEO_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def get_viewpoint_df(scored: pd.DataFrame, strategy: dict, video_meta: dict) -> pd.DataFrame:
    if "viewpoint_df" in st.session_state and not st.session_state.viewpoint_df.empty:
        vp_df = st.session_state.viewpoint_df
        if len(vp_df) == len(scored) and vp.has_viewpoint_columns(vp_df):
            return vp_df
    vp_df = vp.apply_viewpoint_strategy(
        scored,
        strategy,
        video_meta=video_meta,
        english_cache_path=CURRENT_ENGLISH_VIEWPOINT_PATH,
        use_deepseek_english=True,
    )
    st.session_state.viewpoint_df = vp_df
    return vp_df


def get_llm_analysis() -> dict | None:
    for key in ("llm_latest_analysis", "llm_nmf_analysis", "llm_lda_analysis"):
        if key in st.session_state and st.session_state[key]:
            return st.session_state[key]
    if not CURRENT_LLM_ANALYSIS_PATH.exists():
        return None
    try:
        payload = json.loads(CURRENT_LLM_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("version") != vp.ANALYSIS_CACHE_VERSION:
        return None
    analysis = payload.get("analysis", payload) if isinstance(payload, dict) else None
    return analysis if isinstance(analysis, dict) else None


def ensure_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "language",
        "dominant_language",
        "zh_text",
        "en_text",
        "sentiment_score",
        "sentiment_category",
        "sentiment_polarity",
        "zh_sentiment_score",
        "zh_sentiment_category",
        "en_sentiment_score",
        "en_sentiment_category",
    }
    if required.issubset(df.columns):
        return df
    return ta.enrich_dataframe(df)


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color="#d4d8e0", size=12),
        margin=dict(l=14, r=14, t=28, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#c4c8d0")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)", tickfont=dict(color="#b0b4bc"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)", tickfont=dict(color="#b0b4bc"))
    return fig


def daily_chart(df: pd.DataFrame) -> go.Figure:
    daily = df.dropna(subset=["published_at"]).copy()
    if daily.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无时间数据", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 330)

    daily["date"] = daily["published_at"].dt.date
    daily_counts = daily.groupby("date", as_index=False).agg(评论数=("text", "count"), 点赞数=("like_count", "sum"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_counts["date"],
            y=daily_counts["评论数"],
            mode="lines+markers",
            name="评论数",
            line=dict(color=COLORS["teal"], width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Bar(
            x=daily_counts["date"],
            y=daily_counts["点赞数"],
            name="点赞数",
            marker_color="rgba(250,204,21,0.35)",
            marker_line=dict(width=0),
            yaxis="y2",
        )
    )
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=dict(color="#b0b4bc")))
    return style_figure(fig, 330)


def sentiment_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_category"].value_counts().reset_index()
    counts.columns = ["类别", "数量"]
    fig = px.pie(
        counts,
        names="类别",
        values="数量",
        hole=0.55,
        color="类别",
        color_discrete_map={"正面": COLORS["teal"], "中性": COLORS["yellow"], "负面": COLORS["red"]},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", textfont=dict(color="#ffffff", size=13))
    return style_figure(fig, 330)


def polarity_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_polarity"].value_counts().reset_index()
    counts.columns = ["极性", "数量"]
    fig = px.pie(
        counts,
        names="极性",
        values="数量",
        hole=0.55,
        color="极性",
        color_discrete_map={"正向": COLORS["teal"], "负向": COLORS["red"]},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", textfont=dict(color="#ffffff", size=13))
    return style_figure(fig, 330)


def word_bar(tokens: list[str], top_n: int) -> go.Figure:
    words = pd.DataFrame(Counter(tokens).most_common(top_n), columns=["词语", "频次"])
    if words.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无词频数据", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 420)

    words = words.sort_values("频次", ascending=True)
    fig = px.bar(words, x="频次", y="词语", orientation="h", color="频次", color_continuous_scale="Teal")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{x}", textposition="outside", textfont=dict(color="#b0b4bc", size=11))
    return style_figure(fig, 420)


def word_count_bar(word_counts: list, top_n: int) -> go.Figure:
    rows = []
    for item in word_counts[:top_n]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append({"词语": str(item[0]), "频次": int(item[1])})
        elif isinstance(item, dict):
            rows.append({"词语": str(item.get("word", item.get("词语", ""))), "频次": int(item.get("count", item.get("频次", 0)))})
    words = pd.DataFrame(rows)
    if words.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无观点词频数据", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 420)
    words = words.sort_values("频次", ascending=True)
    fig = px.bar(words, x="频次", y="词语", orientation="h", color="频次", color_continuous_scale="Teal")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{x}", textposition="outside", textfont=dict(color="#b0b4bc", size=11))
    return style_figure(fig, 420)


def bilingual_topic_bar(stats: pd.DataFrame, top_n: int) -> go.Figure:
    if stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无双语主题数据", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 420)

    chart_df = stats.head(top_n).sort_values("合并频次", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        x=chart_df["中文频次"],
        y=chart_df["主题词"],
        name="中文词频",
        orientation="h",
        marker_color=COLORS["teal"],
    )
    fig.add_bar(
        x=chart_df["英文频次"],
        y=chart_df["主题词"],
        name="英文映射词频",
        orientation="h",
        marker_color=COLORS["yellow"],
    )
    fig.update_layout(barmode="stack", coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{x}", textposition="outside", textfont=dict(color="#b0b4bc", size=11))
    return style_figure(fig, 420)


def topic_weight_bar(topic_words: list[dict]) -> go.Figure:
    topic_df = pd.DataFrame(topic_words)
    if topic_df.empty or not {"topic", "word", "weight"}.issubset(topic_df.columns):
        fig = go.Figure()
        fig.add_annotation(text="暂无主题模型结果", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 420)
    ordered = topic_df.sort_values(["topic", "weight"], ascending=[True, True])
    fig = px.bar(
        ordered,
        x="weight",
        y="word",
        color="topic",
        facet_col="topic",
        facet_col_wrap=2,
        orientation="h",
        height=max(420, 170 * topic_df["topic"].nunique()),
        title="观点主题词权重",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return style_figure(fig, max(420, 170 * topic_df["topic"].nunique()))


def viewpoint_topic_network_chart(graph_data: dict) -> go.Figure:
    graph = nx.Graph()
    for node in graph_data.get("nodes", []):
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        graph.add_node(
            node_id,
            label=str(node.get("label", node_id)),
            group=str(node.get("group", "观点")),
            size=int(node.get("size", 1)),
            title=str(node.get("title", "")),
        )
    for edge in graph_data.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            graph.add_edge(source, target, weight=float(edge.get("weight", 1)))

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="暂无观点主题网络", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 500)

    positions = nx.spring_layout(graph, seed=42, k=0.9)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    colors = {"观点主题": COLORS["teal"], "观点维度": COLORS["yellow"], "观点词": COLORS["blue"]}
    degrees = dict(graph.degree())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color="rgba(45,212,191,0.20)"),
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
                size=[16 + min(graph.nodes[node].get("size", 1), 34) for node in graph.nodes()],
                color=[colors.get(graph.nodes[node].get("group", ""), COLORS["blue"]) for node in graph.nodes()],
                line=dict(width=1, color=COLORS["bg"]),
            ),
            textfont=dict(color="#e0e4ea", size=11),
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, 500)


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


def semantic_graph(token_lists: list[list[str]], top_n: int, min_edge_weight: int) -> nx.Graph:
    word_counts = Counter(token for tokens in token_lists for token in set(tokens))
    vocabulary = {word for word, _ in word_counts.most_common(top_n)}
    edge_counts: Counter[tuple[str, str]] = Counter()

    for tokens in token_lists:
        unique_tokens = sorted({token for token in tokens if token in vocabulary})
        for index, source in enumerate(unique_tokens):
            for target in unique_tokens[index + 1 :]:
                edge_counts[(source, target)] += 1

    graph = nx.Graph()
    for word in vocabulary:
        graph.add_node(word, size=word_counts[word])
    for (source, target), weight in edge_counts.items():
        if weight >= min_edge_weight:
            graph.add_edge(source, target, weight=weight)
    return graph


def semantic_chart(token_lists: list[list[str]], top_n: int, min_edge_weight: int) -> go.Figure:
    graph = semantic_graph(token_lists, top_n, min_edge_weight)
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="暂无语义网络数据", showarrow=False, font=dict(color="#d4d8e0", size=18))
        return style_figure(fig, 420)

    positions = nx.spring_layout(graph, seed=42, k=0.9)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    degrees = dict(graph.degree())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.0, color="rgba(45,212,191,0.18)"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in graph.nodes()],
            y=[positions[node][1] for node in graph.nodes()],
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            hovertext=[
                f"{node}<br>词频: {graph.nodes[node].get('size', 0)}<br>度: {degrees.get(node, 0)}"
                for node in graph.nodes()
            ],
            hoverinfo="text",
            marker=dict(
                size=[12 + min(graph.nodes[node].get("size", 1), 42) for node in graph.nodes()],
                color=[degrees.get(node, 0) for node in graph.nodes()],
                colorscale=[[0, COLORS["red"]], [0.5, COLORS["yellow"]], [1, COLORS["teal"]]],
                line=dict(width=1, color=COLORS["bg"]),
            ),
            textfont=dict(color="#e0e4ea", size=11),
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, 420)


def metric_card(label: str, value: str, note: str, accent: str = "teal") -> None:
    st.markdown(
        f"""
        <div class="metric-card accent-{html.escape(accent)}">
            <div class="label">{html.escape(label)}</div>
            <div class="value">{html.escape(value)}</div>
            <div class="note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="section-block">
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-desc">{html.escape(desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comments(df: pd.DataFrame, limit: int) -> None:
    comments = df.sort_values("like_count", ascending=False).head(limit)
    if comments.empty:
        st.caption("暂无评论")
        return

    for _, row in comments.iterrows():
        author = html.escape(str(row.get("author", "未知作者"))[:32])
        text = html.escape(str(row.get("text", ""))[:160])
        likes = int(row.get("like_count", 0))
        category = str(row.get("sentiment_category", "中性"))
        language = html.escape(str(row.get("language", "其他")))
        tag_class = {"正面": "positive", "中性": "neutral", "负面": "negative"}.get(category, "neutral")
        st.markdown(
            f"""
            <div class="comment-card">
                <div class="comment-header">
                    <span class="comment-author">{author}</span>
                    <span class="comment-likes">👍 {likes}</span>
                    <span class="comment-likes">{language}</span>
                    <span class="sentiment-tag {tag_class}">{html.escape(category)}</span>
                </div>
                <div class="comment-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_author_table(df: pd.DataFrame, limit: int = 8) -> None:
    author_rank = (
        df.groupby("author", as_index=False)
        .agg(评论数=("text", "count"), 点赞数=("like_count", "sum"))
        .sort_values(["评论数", "点赞数"], ascending=False)
        .head(limit)
    )
    if author_rank.empty:
        st.caption("暂无数据")
        return

    rows_html = ""
    for i, (_, row) in enumerate(author_rank.iterrows()):
        rank = i + 1
        if rank == 1:
            badge_cls = "gold"
        elif rank == 2:
            badge_cls = "silver"
        elif rank == 3:
            badge_cls = "bronze"
        else:
            badge_cls = "normal"
        author_name = html.escape(str(row["author"])[:28])
        rows_html += f"""
        <tr>
            <td><span class="rank-badge {badge_cls}">{rank}</span>{author_name}</td>
            <td>{int(row['评论数'])}</td>
            <td>{int(row['点赞数'])}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="author-table">
            <thead>
                <tr>
                    <th>作者</th>
                    <th>评论数</th>
                    <th>点赞数</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    filtered = df.copy()
    settings = {"原始评论": len(filtered)}

    with st.sidebar:
        st.header("大屏数据源")
        uploaded_file = st.file_uploader("上传评论表格", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                filtered = normalize_dataframe(pd.read_csv(uploaded_file))
            else:
                filtered = normalize_dataframe(pd.read_excel(uploaded_file))
            filtered = ensure_sentiment(filtered)
            settings["原始评论"] = len(filtered)
        else:
            st.caption("未上传时，默认使用主页面当前抓取或上传的评论数据。")

        st.header("分析模块")
        analysis_module = st.radio("大屏模块", ["传统分析", "DeepSeek 大模型分析"], horizontal=True)
        settings["分析模块"] = analysis_module

        st.header("交互筛选")
        if filtered["published_at"].notna().any():
            min_date = filtered["published_at"].min().date()
            max_date = filtered["published_at"].max().date()
            selected_dates = st.date_input("发布时间范围", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                filtered = filtered[
                    filtered["published_at"].isna()
                    | filtered["published_at"].dt.date.between(selected_dates[0], selected_dates[1])
                ]

        analysis_language = st.selectbox("分析语言", ["全部", "中文", "英文"])
        settings["分析语言"] = analysis_language
        if analysis_language == "中文":
            filtered = filtered[filtered["zh_text"].fillna("").astype(str).str.strip().ne("")].copy()
            filtered["sentiment_score"] = filtered["zh_sentiment_score"]
            filtered["sentiment_category"] = filtered["zh_sentiment_category"]
            filtered["sentiment_polarity"] = filtered["sentiment_score"].map(ta.sentiment_polarity)
        elif analysis_language == "英文":
            filtered = filtered[filtered["en_text"].fillna("").astype(str).str.strip().ne("")].copy()
            filtered["sentiment_score"] = filtered["en_sentiment_score"]
            filtered["sentiment_category"] = filtered["en_sentiment_category"]
            filtered["sentiment_polarity"] = filtered["sentiment_score"].map(ta.sentiment_polarity)

        sentiment_options = ["正面", "中性", "负面"]
        selected_sentiments = st.multiselect("情感类别", sentiment_options, default=sentiment_options)
        filtered = filtered[filtered["sentiment_category"].isin(selected_sentiments)]

        max_like = int(filtered["like_count"].max()) if not filtered.empty else 0
        min_like = st.slider("最低点赞数", 0, max(max_like, 1), 0)
        filtered = filtered[filtered["like_count"] >= min_like]

        keyword = st.text_input("评论关键词")
        if keyword.strip():
            filtered = filtered[
                filtered["text"].fillna("").astype(str).str.contains(keyword.strip(), case=False, na=False, regex=False)
            ]

        st.header("图表参数")
        settings["高频词数量"] = st.slider("高频词数量", 8, 40, 18)
        settings["网络词数量"] = st.slider("网络词数量", 20, 120, 46, step=2)
        settings["最小共现"] = st.slider("最小共现次数", 1, 12, 2)
        settings["高赞评论数量"] = st.slider("高赞评论数量", 3, 12, 5)

    settings["筛选后评论"] = len(filtered)
    return filtered, settings


def main() -> None:
    inject_style()
    df = get_source_data()

    st.markdown(
        """
        <div class="hero">
            <div>
                <h1>YouTube 评论舆情数据大屏</h1>
                <p>评论热度、情感结构、主题词与语义共现的实时看板</p>
            </div>
            <div class="badge">Interactive Board · 实时数据</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        with st.sidebar:
            st.header("大屏数据源")
            uploaded_file = st.file_uploader("上传评论表格", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.info("请先回到主页面抓取评论，或在左侧上传包含 text 列的 CSV / Excel。")
            return
        if uploaded_file.name.endswith(".csv"):
            df = normalize_dataframe(pd.read_csv(uploaded_file))
        else:
            df = normalize_dataframe(pd.read_excel(uploaded_file))

    if "text" not in df.columns:
        st.error("表格中需要包含 text 列。")
        return

    scored = ensure_sentiment(normalize_dataframe(df))
    scored, settings = apply_filters(scored)
    if scored.empty:
        st.warning("当前筛选条件下没有评论数据，请调整左侧筛选条件。")
        return

    analysis_language = str(settings.get("分析语言", "全部"))
    zh_token_lists = [ta.tokenize_zh(text) for text in scored["text"].fillna("").astype(str)]
    en_token_lists = [ta.tokenize_en(text) for text in scored["text"].fillna("").astype(str)]
    if analysis_language == "中文":
        token_lists = zh_token_lists
        all_tokens = [token for tokens in zh_token_lists for token in tokens]
    elif analysis_language == "英文":
        token_lists = en_token_lists
        all_tokens = [token for tokens in en_token_lists for token in tokens]
    else:
        token_lists = [zh_tokens + en_tokens for zh_tokens, en_tokens in zip(zh_token_lists, en_token_lists)]
        all_tokens = [token for tokens in token_lists for token in tokens]
    zh_tokens_all = [token for tokens in zh_token_lists for token in tokens]
    en_tokens_all = [token for tokens in en_token_lists for token in tokens]

    date_note = "暂无时间范围"
    if scored["published_at"].notna().any():
        first = scored["published_at"].min().strftime("%Y-%m-%d")
        latest = scored["published_at"].max().strftime("%Y-%m-%d")
        date_note = f"{first} 至 {latest}"

    positive_ratio = scored["sentiment_category"].eq("正面").mean() * 100
    negative_ratio = scored["sentiment_category"].eq("负面").mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("评论总量", f"{len(scored):,}", f"原始 {settings['原始评论']:,} 条", accent="teal")
    with m2:
        metric_card("互动点赞", f"{int(scored['like_count'].sum()):,}", "评论点赞总和", accent="yellow")
    with m3:
        metric_card("参与作者", f"{scored['author'].nunique():,}", "去重作者数量", accent="blue")
    with m4:
        metric_card("正面占比", f"{positive_ratio:.1f}%", f"负面 {negative_ratio:.1f}%", accent="green")
    with m5:
        metric_card("时间范围", date_note, f"筛选后 {settings['筛选后评论']:,} 条", accent="purple")

    mid, right = st.columns(2)
    with mid:
        section_header("情感类别", "中文使用 SnowNLP，英文使用 VADER，统一展示正面/中性/负面")
        st.plotly_chart(sentiment_pie(scored), use_container_width=True)
    with right:
        section_header("情感极性", "将评论分为正向/负向两极，快速判断舆论倾向")
        st.plotly_chart(polarity_pie(scored), use_container_width=True)

    dashboard_module = str(settings.get("分析模块", "传统分析"))
    if dashboard_module == "传统分析":
        lower_left, lower_right = st.columns(2)
        with lower_left:
            if analysis_language == "全部":
                section_header("高频主题词", "中文和英文分开统计词频，避免混合分词影响结果")
                word_zh, word_en = st.columns(2)
                with word_zh:
                    st.plotly_chart(word_bar(zh_tokens_all, int(settings["高频词数量"])), use_container_width=True)
                with word_en:
                    st.plotly_chart(word_bar(en_tokens_all, int(settings["高频词数量"])), use_container_width=True)
            else:
                section_header("高频主题词", f"{analysis_language}分词后统计词频，高频词反映讨论焦点")
                st.plotly_chart(word_bar(all_tokens, int(settings["高频词数量"])), use_container_width=True)
        with lower_right:
            if analysis_language == "全部":
                section_header("语义共现网络", "中文和英文分别构建共现网络，避免跨语言词语误连")
                network_tab_zh, network_tab_en = st.tabs(["中文网络", "英文网络"])
                with network_tab_zh:
                    st.plotly_chart(
                        semantic_chart(zh_token_lists, int(settings["网络词数量"]), int(settings["最小共现"])),
                        use_container_width=True,
                    )
                with network_tab_en:
                    st.plotly_chart(
                        semantic_chart(en_token_lists, int(settings["网络词数量"]), int(settings["最小共现"])),
                        use_container_width=True,
                    )
            else:
                section_header("语义共现网络", f"{analysis_language}词语在同一评论中共同出现则连线，揭示话题关联")
                st.plotly_chart(
                    semantic_chart(token_lists, int(settings["网络词数量"]), int(settings["最小共现"])),
                    use_container_width=True,
                )

        section_header("双语主题合并", "英文核心词翻译或映射为中文主题后，与中文词频合并统计，展示跨语言主题对应关系")
        if not zh_tokens_all or not en_tokens_all:
            st.caption("当前筛选结果缺少中文或英文核心词，无法生成双语主题合并结果。")
        else:
            bilingual_stats = ta.build_bilingual_topic_stats(
                zh_tokens_all,
                en_tokens_all,
                top_n=max(30, int(settings["高频词数量"]) * 2),
            )
            merge_left, merge_right = st.columns([1.15, 1])
            with merge_left:
                st.plotly_chart(
                    bilingual_topic_bar(bilingual_stats, int(settings["高频词数量"])),
                    use_container_width=True,
                )
                st.dataframe(bilingual_stats, use_container_width=True, hide_index=True)
            with merge_right:
                if not ta.wordcloud_font_ready("zh"):
                    st.warning("未找到可用中文字体，无法安全生成双语词云。")
                else:
                    frequencies = ta.bilingual_wordcloud_frequencies(bilingual_stats)
                    if frequencies:
                        st.pyplot(ta.make_wordcloud_from_frequencies(frequencies), use_container_width=True)
                st.caption("英文词已做 NLTK 停用词过滤、手动停用词补充和词形还原；未命中内置词典时会尝试 deep-translator 免费翻译，无需配置密钥。")
    else:
        viewpoint_strategy = get_viewpoint_strategy()
        if not viewpoint_strategy:
            section_header("DeepSeek 大模型分析", "主页面生成观点策略后，大屏会展示观点主题、观点网络、观点词云和双语合并")
            st.caption("当前未检测到 DeepSeek 观点策略。请先在主页面“大模型分析”里加载或生成观点策略。")
        else:
            section_header("DeepSeek 大模型分析", "使用视频专属动态策略，只展示观点、评价、情感、体验和建议类表达")
            viewpoint_df = get_viewpoint_df(scored, viewpoint_strategy, get_video_metadata())
            _, _, viewpoint_token_lists, viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(
                viewpoint_df,
                analysis_language,
                viewpoint_strategy,
            )
            _, _, _, zh_viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(
                viewpoint_df,
                "中文",
                viewpoint_strategy,
            )
            _, _, _, en_viewpoint_tokens = vp.purified_viewpoint_docs_and_tokens(
                viewpoint_df,
                "英文",
                viewpoint_strategy,
            )
            viewpoint_rows = vp.viewpoint_rows_with_purified_tokens(
                viewpoint_df,
                analysis_language,
                viewpoint_strategy,
            )
            llm_analysis = get_llm_analysis()

            st.caption(
                f"视频类型：{viewpoint_strategy.get('video_type', '未知')} | "
                f"核心主题：{viewpoint_strategy.get('core_theme', '评论观点')} | "
                f"观点维度：{' / '.join(str(item) for item in viewpoint_strategy.get('viewpoint_dimensions', []))}"
            )
            if llm_analysis:
                st.caption(
                    f"主题模型缓存：{llm_analysis.get('model', '未知模型')} / "
                    f"{llm_analysis.get('language', '全部')} / {llm_analysis.get('generated_at', '未记录时间')}"
                )
            else:
                st.caption("尚未检测到 DeepSeek 主题模型缓存。主页面运行 LDA 或 NMF 后，大屏会显示自动命名主题和观点网络。")

            vp_metric_1, vp_metric_2, vp_metric_3 = st.columns(3)
            with vp_metric_1:
                metric_card("观点评论", f"{len(viewpoint_rows):,}", f"筛选后 {len(scored):,} 条", accent="teal")
            with vp_metric_2:
                metric_card("观点词", f"{len(viewpoint_tokens):,}", f"{analysis_language}口径", accent="yellow")
            with vp_metric_3:
                metric_card("观点维度", f"{len(viewpoint_strategy.get('viewpoint_dimensions', [])):,}", "DeepSeek 动态生成", accent="blue")

            vp_left, vp_right = st.columns(2)
            with vp_left:
                section_header("观点词频", "只统计 DeepSeek 策略保留的观点词，过滤视频主体名词和无意义高频词")
                if llm_analysis and llm_analysis.get("viewpoint_word_counts"):
                    st.plotly_chart(
                        word_count_bar(llm_analysis.get("viewpoint_word_counts", []), int(settings["高频词数量"])),
                        use_container_width=True,
                    )
                else:
                    st.plotly_chart(word_bar(viewpoint_tokens, int(settings["高频词数量"])), use_container_width=True)
            with vp_right:
                section_header("观点关联网络", "优先展示自动命名主题网络；未生成主题时展示净化后的观点共现")
                graph_data = llm_analysis.get("topic_network", {}) if llm_analysis else {}
                if not graph_data.get("nodes"):
                    graph_data = vp.build_viewpoint_cooccurrence_network(
                        viewpoint_df,
                        viewpoint_strategy,
                        analysis_language,
                    )
                st.plotly_chart(viewpoint_topic_network_chart(graph_data), use_container_width=True)

            section_header("观点主题模型", "显示主页面最后一次 DeepSeek LDA/NMF 结果，包括自动命名主题和词语权重")
            if llm_analysis and llm_analysis.get("topic_words"):
                topic_left, topic_right = st.columns([1.45, 1])
                with topic_left:
                    st.plotly_chart(topic_weight_bar(llm_analysis.get("topic_words", [])), use_container_width=True)
                with topic_right:
                    summaries = pd.DataFrame(llm_analysis.get("topic_summaries", []))
                    st.dataframe(summaries, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无 DeepSeek 主题模型结果。请在主页面的大模型模块运行 LDA 或 NMF。")

            sentiment_left, sentiment_right = st.columns(2)
            with sentiment_left:
                section_header("观点情感类别", "只统计包含观点表达的评论，仍使用中文 SnowNLP、英文 VADER")
                if viewpoint_rows.empty:
                    st.caption("当前筛选下暂无观点评论。")
                else:
                    st.plotly_chart(sentiment_pie(ensure_sentiment(viewpoint_rows)), use_container_width=True)
            with sentiment_right:
                section_header("观点情感极性", "将观点评论统一为正向或负向，辅助判断态度方向")
                if viewpoint_rows.empty:
                    st.caption("当前筛选下暂无观点评论。")
                else:
                    st.plotly_chart(polarity_pie(ensure_sentiment(viewpoint_rows)), use_container_width=True)

            section_header("观点双语合并", "中文观点词和英文观点词保持各自清洗规则，再映射为统一中文主题")
            if not zh_viewpoint_tokens or not en_viewpoint_tokens:
                st.caption("当前筛选结果缺少中文或英文观点词，无法生成观点双语合并结果。")
            else:
                bilingual_stats = ta.build_bilingual_topic_stats(
                    zh_viewpoint_tokens,
                    en_viewpoint_tokens,
                    top_n=max(30, int(settings["高频词数量"]) * 2),
                )
                merge_left, merge_right = st.columns([1.15, 1])
                with merge_left:
                    st.plotly_chart(
                        bilingual_topic_bar(bilingual_stats, int(settings["高频词数量"])),
                        use_container_width=True,
                    )
                    st.dataframe(bilingual_stats, use_container_width=True, hide_index=True)
                with merge_right:
                    if not ta.wordcloud_font_ready("zh"):
                        st.warning("未找到可用中文字体，无法安全生成双语词云。")
                    else:
                        frequencies = ta.bilingual_wordcloud_frequencies(bilingual_stats)
                        if frequencies:
                            st.pyplot(ta.make_wordcloud_from_frequencies(frequencies), use_container_width=True)

    table_left, table_right = st.columns(2)
    with table_left:
        section_header("高赞评论", "按点赞数排序的热门评论，反映最受关注的观点")
        render_comments(scored, settings["高赞评论数量"])
    with table_right:
        section_header("作者互动榜", "评论数和点赞数排名前列的活跃作者")
        render_author_table(scored)


if __name__ == "__main__":
    main()
