from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import jieba
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

try:
    from snownlp import SnowNLP
except ImportError:  # pragma: no cover - requirements include snownlp
    SnowNLP = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - requirements include vaderSentiment
    SentimentIntensityAnalyzer = None

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover - requirements include nltk
    nltk_stopwords = None
    WordNetLemmatizer = None

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - requirements include deep-translator
    GoogleTranslator = None


LanguageCode = Literal["zh", "en", "mixed", "other"]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
EN_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
ZH_TEXT_RE = re.compile(r"[\u4e00-\u9fff]+")

ZH_STOPWORDS = {
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
    "然而", "因此", "此外", "另外", "同时", "并且", "于是", "否则", "无论", "不管",
    "尽管", "既然", "即使", "哪怕", "只要", "只有", "除非", "以便", "以免", "以至于",
    "从而", "进而", "反而", "却", "而", "且", "并", "或", "乃", "亦", "则", "故",
    "让", "把", "被", "给", "对", "向", "往", "从", "在", "到", "于", "为", "与",
    "和", "同", "跟", "比", "按", "照", "据", "凭", "沿", "顺", "逆", "朝", "冲",
    "由", "经", "过", "通过", "关于", "至于", "对于", "鉴于", "基于", "出于", "由于",
    "为了", "用来", "用以", "不了", "为啥", "天哪", "我的天", "嗯嗯", "哦哦", "啊啊",
    "么么", "啦啦", "哈哈哈", "哈哈哈哈",
    "不会", "不能", "不用", "不要", "不想", "不该", "不应", "不会有", "不会是",
    "不是说", "没法", "没用", "没事", "没啥", "没人", "每个", "每次", "每种",
    "各位", "别人", "人家", "某些", "某个", "大家", "人们", "咱们", "咱俩",
    "俺们", "她们", "它们", "这边", "那边", "这里", "那里", "里面", "外面",
    "上面", "下面", "前面", "后面", "之前", "之后", "以内", "以外", "本来",
    "原来", "直接", "基本", "主要", "来说", "来说说", "表示", "看到", "看见",
    "听到", "发现", "认为", "出来", "进去", "出来了", "进去吧",
}

ZH_FUNCTION_WORDS = {
    "不", "没", "很", "挺", "太", "更", "最", "又", "也", "都", "还", "再",
    "才", "就", "只", "别", "请", "会", "能", "要", "想", "看", "说", "讲",
    "问", "来", "去", "有", "没", "是", "和", "与", "及", "或", "并",
}

ZH_KEEP_NEGATIVE_PREFIXES = (
    "不喜欢",
    "不满意",
    "不推荐",
    "不舒服",
    "不好看",
    "不好吃",
    "不好用",
    "不合理",
    "不真实",
    "不清楚",
    "不明白",
    "不值得",
    "不支持",
    "不赞同",
    "不认同",
    "不接受",
    "没用",
    "没意思",
    "没诚意",
    "没必要",
)

EN_STOPWORDS = {
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
    "to", "in", "of", "is", "my", "it", "on", "so", "as", "be", "we", "this",
    "that", "these", "those", "do", "will", "would", "can", "could", "should",
    "may", "might", "must", "need", "going", "getting", "make", "making", "take",
    "took", "taking", "see", "saw", "seeing", "say", "saying", "know", "knew",
    "knowing", "think", "thought", "thinking", "want", "wanted", "wanting",
    "liked", "liking", "love", "loved", "loving", "hate", "hated", "hating",
    "thank", "thankyou", "watch", "watched", "watching", "video", "videos",
    "channel", "comment", "comments", "view", "views", "subscribe", "subscribed",
    "subscribing", "youtube", "youtuber", "youtubers", "yt", "youtubecom",
    "population", "person", "viewer", "viewers", "place", "places", "content",
    "creator", "creators", "mr", "mrs", "ms", "dr", "zhou", "diao", "white",
    "black", "asian", "american", "chinese",
}

EN_OPINION_KEEPWORDS = {
    "good", "great", "bad", "worse", "worst", "best", "better", "love", "loved",
    "loving", "hate", "hated", "hating", "like", "liked", "liking", "amazing",
    "awesome", "beautiful", "boring", "useful", "helpful", "wrong", "true",
    "false", "interesting", "impressive", "disappointing", "disappointed",
}

LANGUAGE_LABELS: dict[LanguageCode, str] = {
    "zh": "中文",
    "en": "英文",
    "mixed": "混合",
    "other": "其他",
}

_VADER_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
_WORDNET_LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None

EN_LEMMA_OVERRIDES = {
    "watched": "watch",
    "watching": "watch",
    "watches": "watch",
    "videos": "video",
    "comments": "comment",
    "views": "view",
    "subscribed": "subscribe",
    "subscribing": "subscribe",
    "used": "use",
    "using": "use",
    "made": "make",
    "making": "make",
    "took": "take",
    "taking": "take",
    "saw": "see",
    "seeing": "see",
    "said": "say",
    "saying": "say",
    "knew": "know",
    "knowing": "know",
    "thought": "think",
    "thinking": "think",
    "wanted": "want",
    "wanting": "want",
    "liked": "like",
    "liking": "like",
    "loved": "love",
    "loving": "love",
    "hated": "hate",
    "hating": "hate",
}

BILINGUAL_TRANSLATION_HINTS = {
    "america": "美国",
    "american": "美国",
    "usa": "美国",
    "us": "美国",
    "china": "中国",
    "chinese": "中国",
    "ccp": "中共",
    "communist": "共产主义",
    "government": "政府",
    "country": "国家",
    "nation": "国家",
    "world": "世界",
    "city": "城市",
    "society": "社会",
    "culture": "文化",
    "history": "历史",
    "policy": "政策",
    "politic": "政治",
    "politics": "政治",
    "democracy": "民主",
    "democratic": "民主",
    "freedom": "自由",
    "liberty": "自由",
    "rights": "权利",
    "human": "人权",
    "war": "战争",
    "military": "军事",
    "army": "军队",
    "weapon": "武器",
    "japan": "日本",
    "japanese": "日本",
    "taiwan": "台湾",
    "hong": "香港",
    "hongkong": "香港",
    "korea": "韩国",
    "russia": "俄罗斯",
    "russian": "俄罗斯",
    "europe": "欧洲",
    "european": "欧洲",
    "economy": "经济",
    "economic": "经济",
    "money": "经济",
    "technology": "科技",
    "infrastructure": "基建",
    "education": "教育",
    "school": "教育",
    "student": "学生",
    "life": "生活",
    "food": "食物",
    "travel": "旅行",
    "tourism": "旅游",
    "ice": "冰雪",
    "snow": "冰雪",
    "winter": "冬季",
    "harbin": "哈尔滨",
    "truth": "真相",
    "news": "新闻",
    "media": "媒体",
    "internet": "互联网",
    "youtube": "油管",
    "police": "警察",
    "law": "法律",
    "power": "权力",
    "party": "政党",
}


def strip_noise(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"@\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_language_text(text: str) -> tuple[str, str]:
    cleaned = strip_noise(text)
    zh_text = " ".join(ZH_TEXT_RE.findall(cleaned))
    en_text = " ".join(match.group(0) for match in EN_TOKEN_RE.finditer(cleaned.lower()))
    return zh_text, en_text


def detect_language(text: str) -> LanguageCode:
    cleaned = strip_noise(text)
    has_zh = bool(CJK_RE.search(cleaned))
    has_en = bool(EN_RE.search(cleaned))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "other"


def dominant_language(text: str) -> LanguageCode:
    zh_text, en_text = split_language_text(text)
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", zh_text))
    en_count = len(EN_TOKEN_RE.findall(en_text))
    if zh_count == 0 and en_count == 0:
        return "other"
    return "zh" if zh_count >= en_count else "en"


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, "其他")


def clean_text(text: str) -> str:
    text = strip_noise(text)
    text = re.sub(r"[^\w\u4e00-\u9fff']+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    words = set(EN_STOPWORDS)
    if nltk_stopwords is not None:
        try:
            words.update(nltk_stopwords.words("english"))
        except LookupError:
            pass
    words.difference_update(EN_OPINION_KEEPWORDS)
    return frozenset(words)


def simple_english_lemma(token: str) -> str:
    if token in EN_LEMMA_OVERRIDES:
        return EN_LEMMA_OVERRIDES[token]
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "ses", "zes")):
        return token[:-2]
    if len(token) > 4 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) <= 3:
            return token
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem in {"mak", "tak", "us"}:
            return {"mak": "make", "tak": "take", "us": "use"}[stem]
        return stem
    if len(token) > 3 and token.endswith("ed"):
        stem = token[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def lemmatize_english_token(token: str) -> str:
    token = token.lower().strip("'")
    if token in EN_LEMMA_OVERRIDES:
        return EN_LEMMA_OVERRIDES[token]
    if _WORDNET_LEMMATIZER is None:
        return simple_english_lemma(token)
    try:
        lemma = _WORDNET_LEMMATIZER.lemmatize(token, "n")
        lemma = _WORDNET_LEMMATIZER.lemmatize(lemma, "v")
        lemma = _WORDNET_LEMMATIZER.lemmatize(lemma, "a")
        return lemma
    except LookupError:
        return simple_english_lemma(token)


def is_strict_zh_noise(token: str) -> bool:
    token = token.strip()
    if token in ZH_KEEP_NEGATIVE_PREFIXES:
        return False
    if not token or token in ZH_STOPWORDS or token in ZH_FUNCTION_WORDS:
        return True
    if len(token) < 2 or token.isdigit():
        return True
    if token.startswith(("他们", "她们", "你们", "我们", "这些", "那些", "这种", "那种", "这个", "那个")):
        return True
    if token.startswith(("不会", "不能", "不用", "不要", "不想", "不该", "不应")):
        return not token.startswith(ZH_KEEP_NEGATIVE_PREFIXES)
    if token in {"还是", "还有", "或者", "而且", "所以", "因为", "但是", "如果", "然后"}:
        return True
    return False


def tokenize_zh(text: str) -> list[str]:
    zh_text, _ = split_language_text(text)
    protected_tokens: list[str] = []
    for phrase in sorted(ZH_KEEP_NEGATIVE_PREFIXES, key=len, reverse=True):
        while phrase in zh_text:
            protected_tokens.append(phrase)
            zh_text = zh_text.replace(phrase, " ", 1)
    tokens: list[str] = []
    for token in protected_tokens:
        if not is_strict_zh_noise(token):
            tokens.append(token)
    for token in jieba.lcut(zh_text):
        token = token.strip()
        if is_strict_zh_noise(token):
            continue
        tokens.append(token)
    return tokens


def tokenize_en(text: str) -> list[str]:
    _, en_text = split_language_text(text)
    stopwords = english_stopwords()
    tokens: list[str] = []
    for match in EN_TOKEN_RE.finditer(en_text.lower()):
        raw_token = match.group(0).strip("'")
        token = lemmatize_english_token(raw_token)
        if raw_token in stopwords or token in stopwords:
            continue
        if len(token) < 2 or token.isdigit():
            continue
        tokens.append(token)
    return tokens


def tokenize_by_language(text: str, language: Literal["zh", "en"]) -> list[str]:
    return tokenize_zh(text) if language == "zh" else tokenize_en(text)


def tokenize(text: str) -> list[str]:
    return tokenize_zh(text) + tokenize_en(text)


def prepared_documents(texts: Iterable[str], language: Literal["zh", "en"] | None = None) -> list[str]:
    if language is None:
        return [" ".join(tokenize(text)) for text in texts]
    return [" ".join(tokenize_by_language(text, language)) for text in texts]


def sentiment_label(score: float) -> str:
    if score >= 0.6:
        return "正面"
    if score <= 0.4:
        return "负面"
    return "中性"


def sentiment_polarity(score: float) -> str:
    return "正向" if score >= 0.5 else "负向"


def chinese_sentiment_score(text: str) -> float:
    zh_text, _ = split_language_text(text)
    if SnowNLP is None or not zh_text.strip():
        return 0.5
    try:
        return float(SnowNLP(zh_text).sentiments)
    except Exception:
        return 0.5


def english_sentiment_score(text: str) -> float:
    _, en_text = split_language_text(text)
    if _VADER_ANALYZER is None or not en_text.strip():
        return 0.5
    compound = float(_VADER_ANALYZER.polarity_scores(en_text)["compound"])
    return (compound + 1) / 2


def sentiment_score(text: str) -> float:
    dominant = dominant_language(text)
    if dominant == "zh":
        return chinese_sentiment_score(text)
    if dominant == "en":
        return english_sentiment_score(text)
    return 0.5


def analyze_comment(text: str) -> dict[str, object]:
    zh_text, en_text = split_language_text(text)
    detected = detect_language(text)
    dominant = dominant_language(text)
    score = sentiment_score(text)
    zh_score = chinese_sentiment_score(text) if zh_text else 0.5
    en_score = english_sentiment_score(text) if en_text else 0.5
    return {
        "language": language_label(detected),
        "dominant_language": language_label(dominant),
        "zh_text": zh_text,
        "en_text": en_text,
        "sentiment_score": score,
        "sentiment_category": sentiment_label(score),
        "sentiment_polarity": sentiment_polarity(score),
        "zh_sentiment_score": zh_score,
        "zh_sentiment_category": sentiment_label(zh_score),
        "en_sentiment_score": en_score,
        "en_sentiment_category": sentiment_label(en_score),
    }


def enrich_dataframe(df):
    enriched = df.copy()
    if "text" not in enriched.columns:
        enriched["text"] = ""
    analysis = enriched["text"].fillna("").astype(str).map(analyze_comment).apply(dict)
    analysis_df = type(enriched)(analysis.tolist(), index=enriched.index)
    for column in analysis_df.columns:
        enriched[column] = analysis_df[column]
    return enriched


def find_font(language: Literal["zh", "en"] = "zh") -> str | None:
    if language == "en":
        return None
    preferred = [
        "Arial Unicode.ttf",
        "Arial Unicode MS.ttf",
        "PingFang.ttc",
        "Songti.ttc",
        "Hiragino Sans GB.ttc",
        "NotoSansCJK-Regular.ttc",
        "Noto Sans CJK",
        "SimHei.ttf",
        "Microsoft YaHei.ttf",
        "WenQuanYi",
    ]
    for font_path in font_manager.findSystemFonts():
        font_name = Path(font_path).name.lower()
        if any(name.lower() in font_name for name in preferred):
            return font_path
    return None


def make_wordcloud(tokens: list[str], language: Literal["zh", "en"] = "zh") -> plt.Figure:
    font_path = find_font(language)
    wc = WordCloud(
        width=1400,
        height=760,
        background_color="white",
        colormap="Set2" if language == "zh" else "tab20",
        font_path=font_path,
        max_words=180,
        collocations=False,
        prefer_horizontal=0.9,
        regexp=None,
    ).generate(" ".join(tokens))
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def wordcloud_font_ready(language: Literal["zh", "en"]) -> bool:
    return language == "en" or find_font("zh") is not None


@lru_cache(maxsize=1024)
def translate_english_term(term: str) -> tuple[str, str]:
    normalized = lemmatize_english_token(term)
    if normalized in BILINGUAL_TRANSLATION_HINTS:
        return BILINGUAL_TRANSLATION_HINTS[normalized], "内置词典"

    if GoogleTranslator is None:
        return normalized, "英文原词"

    try:
        translated = GoogleTranslator(source="en", target="zh-CN").translate(normalized)
    except Exception:
        return normalized, "英文原词"

    translated = str(translated or "").strip()
    return (translated, "免费翻译") if translated else (normalized, "英文原词")


def canonical_topic_label(translated: str, zh_counts: Counter[str]) -> str:
    translated = translated.strip().lower()
    cjk_parts = re.findall(r"[\u4e00-\u9fff]+", translated)
    if cjk_parts:
        zh_label = "".join(cjk_parts)
        for zh_word, _ in zh_counts.most_common():
            if zh_word == zh_label or zh_word in zh_label or zh_label in zh_word:
                return zh_word
        return zh_label[:12]
    return translated


def build_bilingual_topic_stats(
    zh_tokens: list[str],
    en_tokens: list[str],
    top_n: int = 40,
    core_pool_size: int = 80,
    max_free_translations: int = 30,
) -> pd.DataFrame:
    zh_counts = Counter(zh_tokens)
    en_counts = Counter(en_tokens)
    records: dict[str, dict[str, object]] = {}

    def ensure_record(topic: str) -> dict[str, object]:
        if topic not in records:
            records[topic] = {
                "主题词": topic,
                "合并频次": 0,
                "中文频次": 0,
                "英文频次": 0,
                "中文核心词": Counter(),
                "英文核心词": Counter(),
                "英文映射": [],
                "映射方式": set(),
            }
        return records[topic]

    for zh_word, count in zh_counts.most_common(core_pool_size):
        record = ensure_record(zh_word)
        record["中文频次"] = int(record["中文频次"]) + count
        record["合并频次"] = int(record["合并频次"]) + count
        record["中文核心词"][zh_word] += count

    free_translation_attempts = 0
    for en_word, count in en_counts.most_common(core_pool_size):
        normalized = lemmatize_english_token(en_word)
        if normalized in BILINGUAL_TRANSLATION_HINTS:
            translated, method = translate_english_term(normalized)
        elif free_translation_attempts < max_free_translations:
            free_translation_attempts += 1
            translated, method = translate_english_term(normalized)
        else:
            translated, method = normalized, "英文原词"
        topic = canonical_topic_label(translated, zh_counts)
        record = ensure_record(topic)
        record["英文频次"] = int(record["英文频次"]) + count
        record["合并频次"] = int(record["合并频次"]) + count
        record["英文核心词"][en_word] += count
        record["英文映射"].append(f"{en_word} -> {topic}")
        record["映射方式"].add(method)

    rows = []
    for record in records.values():
        zh_terms = record["中文核心词"].most_common(4)
        en_terms = record["英文核心词"].most_common(4)
        rows.append(
            {
                "主题词": record["主题词"],
                "合并频次": int(record["合并频次"]),
                "中文频次": int(record["中文频次"]),
                "英文频次": int(record["英文频次"]),
                "中文核心词": "、".join(f"{word}({count})" for word, count in zh_terms),
                "英文核心词": ", ".join(f"{word}({count})" for word, count in en_terms),
                "英文映射": "；".join(record["英文映射"][:4]),
                "映射方式": "、".join(sorted(record["映射方式"])) if record["映射方式"] else "中文原词",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["主题词", "合并频次", "中文频次", "英文频次", "中文核心词", "英文核心词", "英文映射", "映射方式"])

    stats = pd.DataFrame(rows)
    stats = stats.sort_values(["合并频次", "英文频次", "中文频次"], ascending=False)
    return stats.head(top_n).reset_index(drop=True)


def bilingual_wordcloud_frequencies(stats: pd.DataFrame) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for _, row in stats.iterrows():
        label = str(row.get("主题词", "")).strip()
        english_terms = str(row.get("英文核心词", "")).strip()
        if english_terms:
            english_head = english_terms.split("(")[0].split(",")[0].strip()
            if english_head and english_head != label:
                label = f"{label} {english_head}"
        if label:
            frequencies[label] = int(row.get("合并频次", 0))
    return frequencies


def make_wordcloud_from_frequencies(frequencies: dict[str, int]) -> plt.Figure:
    wc = WordCloud(
        width=1400,
        height=760,
        background_color="white",
        colormap="Set2",
        font_path=find_font("zh"),
        max_words=180,
        collocations=False,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig
