from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

import text_analysis as ta


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

STRATEGY_COLUMNS = [
    "zh_viewpoint_text",
    "en_viewpoint_text",
    "viewpoint_text",
    "zh_viewpoint_tokens",
    "en_viewpoint_tokens",
    "viewpoint_tokens",
    "zh_anchor_tokens",
    "en_anchor_tokens",
    "anchor_tokens",
    "zh_viewpoint_filter_mode",
    "en_viewpoint_filter_mode",
    "zh_viewpoint_fallback_level",
    "en_viewpoint_fallback_level",
    "en_pivot_zh_viewpoint_tokens",
    "en_aligned_viewpoint_tokens",
    "en_viewpoint_alignments",
    "en_viewpoint_extraction_mode",
]

ANALYSIS_CACHE_VERSION = 3
ENGLISH_VIEWPOINT_CACHE_VERSION = 2

ZH_OPINION_SEEDS = {
    "喜欢", "不喜欢", "满意", "不满意", "推荐", "不推荐", "失望", "期待", "希望",
    "改进", "支持", "反对", "赞同", "不赞同", "认同", "不认同", "好看", "不好看",
    "好吃", "不好吃", "好用", "不好用", "有用", "没用", "清楚", "不清楚", "真实",
    "不真实", "片面", "偏见", "无聊", "有趣", "震撼", "感动", "愤怒", "离谱",
    "合理", "不合理", "值得", "不值得", "舒服", "不舒服", "开心", "难过",
    "好评", "差评", "吐槽", "建议", "担心", "质疑", "怀疑", "反感", "认可", "赞赏",
    "感人", "感动", "震惊", "震撼", "失望", "遗憾", "尴尬", "荒唐", "可笑", "讽刺",
    "精彩", "糟糕", "优秀", "一般", "普通", "惊喜", "无感", "共鸣", "反驳", "误导",
    "夸张", "客观", "主观", "浅薄", "深刻", "清晰", "混乱", "舒服", "不适", "难受",
    "接受", "不接受", "尊重", "不尊重", "可信", "不可信", "喜欢看", "值得看",
    "讲得清楚", "讲不清楚", "太片面", "有帮助", "没帮助", "很失望", "希望改进",
}

EN_OPINION_SEEDS = {
    "good", "great", "bad", "terrible", "boring", "useful", "helpful", "useless",
    "wrong", "true", "fake", "biased", "racist", "amazing", "awesome", "beautiful",
    "disappointed", "disappointing", "support", "oppose", "agree", "disagree",
    "recommend", "confused", "clear", "unclear", "funny", "sad", "angry", "hope",
    "expect", "love", "hate", "like", "interesting", "impressive", "one-sided",
    "misleading", "honest", "valuable", "ridiculous", "unfair", "fair",
    "better", "best", "worse", "worst", "excellent", "poor", "awful", "nice",
    "cool", "weak", "strong", "accurate", "inaccurate", "real", "false", "touching",
    "moving", "shocked", "shocking", "surprised", "surprising", "annoying", "annoyed",
    "disrespectful", "respectful", "objective", "subjective", "shallow", "deep",
    "valuable", "meaningful", "meaningless", "overrated", "underrated", "exaggerated",
    "reasonable", "unreasonable", "credible", "unbelievable", "trustworthy",
    "untrustworthy", "agreeable", "painful", "comfortable", "uncomfortable",
}

ZH_BASE_DROP_TERMS = {
    "他们", "她们", "我们", "你们", "这个", "那个", "这些", "那些", "这种", "那种",
    "因为", "所以", "但是", "如果", "然后", "而且", "虽然", "不过", "越来越",
    "看到", "觉得", "认为", "表示", "知道", "视频", "评论", "作者", "博主",
    "不会", "不能", "不用", "不要", "不是", "没有", "还有", "其实", "只是", "一样",
    "一下", "一直", "已经", "现在", "可能", "应该", "什么", "怎么", "这样", "那样",
    "大家", "人们", "别人", "人家", "时候", "地方", "东西", "事情", "问题", "情况",
}

EN_BASE_DROP_TERMS = {
    "youtube", "youtuber", "yt", "video", "videos", "channel", "comment", "comments",
    "subscribe", "watch", "watched", "view", "views", "people", "person", "population",
    "country", "world", "thing", "stuff", "mr", "mrs", "ms", "dr", "zhou", "diao",
    "white", "black", "asian", "chinese", "american",
    "youtubecom", "youtubers", "viewer", "viewers", "someone", "everyone", "anyone",
    "place", "places", "name", "names", "man", "woman", "men", "women", "kid", "kids",
    "guy", "guys", "stuff", "content", "creator", "creators", "episode", "short",
}

EN_NEGATION_WORDS = {"not", "no", "never", "without", "hardly"}


def default_strategy() -> dict[str, Any]:
    return {
        "video_type": "未知",
        "core_theme": "评论观点",
        "viewpoint_dimensions": ["评价", "情绪", "体验", "建议", "争议"],
        "zh": {"drop_terms": [], "viewpoint_terms": [], "anchor_terms": [], "aliases": {}},
        "en": {"drop_terms": [], "viewpoint_terms": [], "anchor_terms": [], "aliases": {}},
    }


def normalize_word_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[,，、;\s]+", value)
    elif isinstance(value, list):
        items = value
    else:
        items = []
    normalized = []
    seen = set()
    for item in items:
        word = str(item).strip().lower()
        if not word or word in seen:
            continue
        seen.add(word)
        normalized.append(word)
    return normalized


def normalize_aliases(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    aliases: dict[str, str] = {}
    for key, alias in value.items():
        key_text = str(key).strip().lower()
        alias_text = str(alias).strip().lower()
        if key_text and alias_text:
            aliases[key_text] = alias_text
    return aliases


def merge_unique(*groups: Any) -> list[str]:
    merged: list[str] = []
    seen = set()
    for group in groups:
        for item in normalize_word_list(group):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def normalize_strategy(strategy: dict[str, Any] | None) -> dict[str, Any]:
    base = default_strategy()
    if isinstance(strategy, dict):
        base["video_type"] = str(strategy.get("video_type") or base["video_type"]).strip()[:40]
        base["core_theme"] = str(strategy.get("core_theme") or base["core_theme"]).strip()[:80]
        base["viewpoint_dimensions"] = normalize_word_list(strategy.get("viewpoint_dimensions")) or base[
            "viewpoint_dimensions"
        ]
        for lang in ("zh", "en"):
            source = strategy.get(lang) if isinstance(strategy.get(lang), dict) else {}
            base[lang] = {
                "drop_terms": normalize_word_list(source.get("drop_terms")),
                "viewpoint_terms": normalize_word_list(source.get("viewpoint_terms")),
                "anchor_terms": normalize_word_list(source.get("anchor_terms")),
                "aliases": normalize_aliases(source.get("aliases")),
            }
    return base


def strategy_cache_key(video_meta: dict[str, Any], comments: list[str], zh_candidates: list[str], en_candidates: list[str]) -> str:
    payload = {
        "title": str(video_meta.get("title", "")),
        "description": str(video_meta.get("description", ""))[:2000],
        "comments": comments[:80],
        "zh_candidates": zh_candidates[:160],
        "en_candidates": en_candidates[:160],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_cached_strategy(cache_path: Path, cache_key: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    return normalize_strategy(payload.get("strategy"))


def write_cached_strategy(cache_path: Path, cache_key: str, strategy: dict[str, Any]) -> None:
    cache_path.parent.mkdir(exist_ok=True)
    payload = {"cache_key": cache_key, "strategy": normalize_strategy(strategy)}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cached_analysis(cache_path: Path, cache_key: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != cache_key or payload.get("version") != ANALYSIS_CACHE_VERSION:
        return None
    analysis = payload.get("analysis")
    return analysis if isinstance(analysis, dict) else None


def write_cached_analysis(cache_path: Path, cache_key: str, analysis: dict[str, Any]) -> None:
    cache_path.parent.mkdir(exist_ok=True)
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "cache_key": cache_key,
        "analysis": analysis,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_analysis_cache_key(video_meta: dict[str, Any], strategy: dict[str, Any], df: pd.DataFrame) -> str:
    comments = sample_comments(df, max_comments=120)
    payload = {
        "video": {
            "title": str(video_meta.get("title", "")),
            "description": str(video_meta.get("description", ""))[:2000],
        },
        "strategy": normalize_strategy(strategy),
        "comments": comments,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_english_viewpoint_cache_key(
    video_meta: dict[str, Any],
    strategy: dict[str, Any],
    comments: list[str],
) -> str:
    payload = {
        "version": ENGLISH_VIEWPOINT_CACHE_VERSION,
        "video": {
            "title": str(video_meta.get("title", "")),
            "description": str(video_meta.get("description", ""))[:2000],
        },
        "strategy": normalize_strategy(strategy),
        "comments": [str(comment) for comment in comments],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_cached_english_viewpoints(cache_path: Path, cache_key: str) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != cache_key or payload.get("version") != ENGLISH_VIEWPOINT_CACHE_VERSION:
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def write_cached_english_viewpoints(cache_path: Path, cache_key: str, rows: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(exist_ok=True)
    payload = {
        "version": ENGLISH_VIEWPOINT_CACHE_VERSION,
        "cache_key": cache_key,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("大模型未返回 JSON 对象。")
    return json.loads(match.group(0))


def deepseek_chat_json(
    api_key: str,
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("未配置 DeepSeek API Key。")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            last_error = exc
            if status_code and status_code not in {408, 409, 425, 429, 502, 503, 504} and status_code < 500:
                break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            last_error = exc
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(3.0 * attempt, 10.0))

    raise RuntimeError(f"DeepSeek 请求失败（已重试 {retries} 次）：{last_error}")


def candidate_terms(texts: list[str], language: Literal["zh", "en"], top_n: int = 160) -> list[str]:
    tokens: list[str] = []
    tokenizer = ta.tokenize_zh if language == "zh" else ta.tokenize_en
    for text in texts:
        tokens.extend(tokenizer(text))
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def is_noise_token(token: str, language: Literal["zh", "en"], strategy: dict[str, Any]) -> bool:
    token = str(token).strip().lower()
    if not token:
        return True
    if language == "en" and "_" in token:
        parts = [part for part in token.split("_") if part]
        if parts:
            return all(part in EN_NEGATION_WORDS or is_noise_token(part, language, strategy) for part in parts)
    lang_strategy = strategy[language]
    if token in set(normalize_word_list(lang_strategy.get("drop_terms"))):
        return True
    if token in set(normalize_word_list(lang_strategy.get("anchor_terms"))):
        return True
    if language == "zh":
        return ta.is_strict_zh_noise(token)
    stopwords = ta.english_stopwords()
    return token in stopwords or token in EN_BASE_DROP_TERMS or len(token) < 3


def is_opinion_like_token(token: str, language: Literal["zh", "en"], strategy: dict[str, Any]) -> bool:
    token = str(token).strip().lower()
    if is_noise_token(token, language, strategy):
        return False
    if language == "en" and "_" in token:
        parts = [part for part in token.split("_") if part and part not in EN_NEGATION_WORDS]
        return bool(parts) and any(is_opinion_like_token(part, language, strategy) for part in parts)
    viewpoint_terms = expanded_viewpoint_terms(strategy[language], language)
    if token in viewpoint_terms:
        return True
    if language == "zh":
        return token in ZH_OPINION_SEEDS or token.startswith(("不", "没")) or token.endswith(("好", "差", "棒", "烂"))
    return token in EN_OPINION_SEEDS or token.endswith(("ful", "less", "ing", "ed", "able", "ive"))


def build_dynamic_drop_terms(texts: list[str], strategy: dict[str, Any], language: Literal["zh", "en"]) -> list[str]:
    tokenizer = ta.tokenize_zh if language == "zh" else ta.tokenize_en
    doc_counts: Counter[str] = Counter()
    total_docs = 0
    for text in texts:
        tokens = set(tokenizer(text))
        if not tokens:
            continue
        total_docs += 1
        doc_counts.update(tokens)
    if total_docs == 0:
        return []

    dynamic_drop: list[str] = []
    for token, count in doc_counts.most_common(160):
        doc_ratio = count / total_docs
        if language == "zh":
            forced = token in ZH_BASE_DROP_TERMS or ta.is_strict_zh_noise(token)
        else:
            forced = token in EN_BASE_DROP_TERMS or token in ta.english_stopwords()
        if forced:
            dynamic_drop.append(token)
            continue
        if doc_ratio >= 0.08 and not is_opinion_like_token(token, language, strategy):
            dynamic_drop.append(token)
    return dynamic_drop


def enhance_strategy_for_texts(strategy: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    enhanced = normalize_strategy(strategy)
    zh_dynamic_drop = build_dynamic_drop_terms(texts, enhanced, "zh")
    en_dynamic_drop = build_dynamic_drop_terms(texts, enhanced, "en")
    enhanced["zh"]["drop_terms"] = merge_unique(enhanced["zh"]["drop_terms"], ZH_BASE_DROP_TERMS, zh_dynamic_drop)
    enhanced["en"]["drop_terms"] = merge_unique(enhanced["en"]["drop_terms"], EN_BASE_DROP_TERMS, en_dynamic_drop)
    enhanced["zh"]["viewpoint_terms"] = merge_unique(enhanced["zh"]["viewpoint_terms"], ZH_OPINION_SEEDS)
    enhanced["en"]["viewpoint_terms"] = merge_unique(enhanced["en"]["viewpoint_terms"], EN_OPINION_SEEDS)
    return enhanced


def sample_comments(df: pd.DataFrame, max_comments: int = 80) -> list[str]:
    if df.empty or "text" not in df.columns:
        return []
    sampled = df.copy()
    if "like_count" in sampled.columns:
        sampled["like_count"] = pd.to_numeric(sampled["like_count"], errors="coerce").fillna(0)
        sampled = sampled.sort_values("like_count", ascending=False)
    comments = []
    for text in sampled["text"].fillna("").astype(str):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        comments.append(text[:240])
        if len(comments) >= max_comments:
            break
    return comments


def strategy_prompt(
    video_meta: dict[str, Any],
    comments: list[str],
    zh_candidates: list[str],
    en_candidates: list[str],
    max_description_chars: int = 3000,
) -> list[dict[str, str]]:
    title = str(video_meta.get("title", "")).strip()
    description = str(video_meta.get("description", "")).strip()[:max_description_chars]
    channel = str(video_meta.get("channel_title", "")).strip()
    tags = video_meta.get("tags", [])
    if isinstance(tags, list):
        tags_text = "、".join(str(tag) for tag in tags[:30])
    else:
        tags_text = str(tags)

    user_payload = {
        "video": {
            "title": title,
            "description": description,
            "channel": channel,
            "tags": tags_text,
        },
        "comment_samples": comments,
        "zh_candidate_terms": zh_candidates,
        "en_candidate_terms": en_candidates,
    }
    schema = {
        "video_type": "视频类型，动态判断，不限于固定分类",
        "core_theme": "视频核心主题，一句话",
        "viewpoint_dimensions": ["评论中值得观察的观点维度"],
        "zh": {
            "drop_terms": ["本视频语境下无业务价值的高频词、主体名词、地名、泛名词"],
            "viewpoint_terms": ["中文观点/评价/情感/体验/建议词"],
            "anchor_terms": ["可作为语境锚点但不应主导词频和网络的词"],
            "aliases": {"原词": "归一后的观点词"},
        },
        "en": {
            "drop_terms": ["English noisy nouns/entity words in this video context"],
            "viewpoint_terms": ["English opinion/evaluation/emotion/experience/suggestion terms"],
            "anchor_terms": ["English context anchors not used as central opinion nodes"],
            "aliases": {"raw term": "canonical opinion term"},
        },
    }
    system = (
        "你是舆情分析策略生成器。你必须根据当前视频标题、描述和评论样本动态判断，"
        "不要套用固定行业词表或固定分类。你的任务是让后续词频、LDA/NMF、共现网络聚焦用户观点，"
        "剔除视频主体名词、地点、人物、频道名、泛泛名词和无意义高频词。只返回合法 JSON。"
    )
    user = (
        "请根据输入生成该视频专属的数据清洗策略。要求：\n"
        "1. 识别视频类型、核心主题、评论应关注的观点维度。\n"
        "2. 中英文分别给出 drop_terms、viewpoint_terms、anchor_terms、aliases。\n"
        "3. viewpoint_terms 只保留观点/评价/情感/体验/建议/争议表达；anchor_terms 可解释上下文但不进入主网络。\n"
        "4. viewpoint_terms 必须足量且有区分度：中文不少于 60 个，英文不少于 80 个；"
        "必须覆盖正面评价、负面评价、情绪、体验、建议/期待、争议/反驳六类。"
        "不要只输出名词，必须包含形容词、动词、短语和否定表达。\n"
        "5. 英文 viewpoint_terms 应包含词形变体和短语，如 boring, bored, helpful, useless, biased, agree, disagree, makes sense, not true。"
        "中文 viewpoint_terms 应包含短语，如 不好看、不喜欢、很失望、值得看、讲得清楚、太片面、希望改进。\n"
        "6. 不要把视频主体、地名、频道名、人物名、国家名、普通名词放进 viewpoint_terms。\n"
        "7. drop_terms 必须主动覆盖代词、连接词、平台词、高频泛名词、人名、地名、视频主体名词。\n"
        "8. 输出 JSON 必须符合这个结构：\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"输入：\n{json.dumps(user_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_viewpoint_strategy(
    df: pd.DataFrame,
    video_meta: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> tuple[dict[str, Any], bool]:
    texts = df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    comments = sample_comments(df)
    zh_candidates = candidate_terms(texts, "zh")
    en_candidates = candidate_terms(texts, "en")
    cache_key = strategy_cache_key(video_meta, comments, zh_candidates, en_candidates)

    if cache_path and not force_refresh:
        cached = read_cached_strategy(cache_path, cache_key)
        if cached is not None:
            return enhance_strategy_for_texts(cached, texts), True

    if not api_key:
        raise RuntimeError("未找到匹配的观点策略缓存，且未配置 DeepSeek API Key。")

    prompt_attempts = [
        (comments, zh_candidates, en_candidates, 3000, 0.2),
        (comments[:50], zh_candidates[:120], en_candidates[:120], 1800, 0.1),
        (comments[:25], zh_candidates[:70], en_candidates[:70], 900, 0.0),
    ]
    last_error: Exception | None = None
    raw_strategy: dict[str, Any] | None = None
    for attempt_comments, attempt_zh, attempt_en, desc_limit, temp in prompt_attempts:
        messages = strategy_prompt(
            video_meta,
            attempt_comments,
            attempt_zh,
            attempt_en,
            max_description_chars=desc_limit,
        )
        try:
            raw_strategy = deepseek_chat_json(
                api_key,
                messages,
                model=model,
                temperature=temp,
                timeout=180,
                retries=3,
            )
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2)

    if raw_strategy is None:
        raise RuntimeError(f"DeepSeek 观点策略生成失败：{last_error}")

    strategy = enhance_strategy_for_texts(normalize_strategy(raw_strategy), texts)
    if cache_path:
        write_cached_strategy(cache_path, cache_key, strategy)
    return strategy, False


def english_viewpoint_prompt(
    video_meta: dict[str, Any],
    strategy: dict[str, Any],
    comments: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "video": {
            "title": str(video_meta.get("title", "")),
            "description": str(video_meta.get("description", ""))[:1500],
        },
        "strategy": normalize_strategy(strategy),
        "comments": comments,
    }
    schema = {
        "comments": [
            {
                "comment_index": 0,
                "zh_viewpoint_terms": ["翻译成中文后的观点/评价/情感/体验/建议词"],
                "en_viewpoint_terms": ["原英文评论中语义对应的英文观点词或短语"],
                "alignments": [
                    {
                        "zh": "中文观点词",
                        "en": "英文原词或短语",
                        "dimension": "最贴近的观点维度",
                        "polarity": "positive/negative/neutral",
                        "confidence": 0.9,
                    }
                ],
                "drop_terms": ["本评论中不应进入观点分析的英文噪声词"],
            }
        ]
    }
    system = (
        "你是双语舆情观点抽取器。你必须逐条阅读英文评论，先理解并翻译出中文观点词，"
        "再回到原英文评论中找语义对应的英文原词或短语。只保留观点、评价、情感、体验、建议、争议表达。"
        "不要输出 YouTube、video、channel、comment、country、place、person、brand 等主体名词或平台词，"
        "除非它们本身是明确评价词的一部分。只返回合法 JSON。"
    )
    user = (
        "请按 comment_index 返回每条评论的中英文观点词对齐结果。要求：\n"
        "1. zh_viewpoint_terms 是该英文评论翻译和理解后的中文观点词，不要放代词、助词、平台词、普通名词。\n"
        "2. en_viewpoint_terms 必须来自原英文评论中的词或短语；不能凭空创造英文表达。\n"
        "3. alignments 用于说明中文观点词和英文原词/短语的对应关系，并标注 dimension 与 polarity。\n"
        "4. dimension 优先从输入 strategy.viewpoint_dimensions 中选择；polarity 只能是 positive、negative、neutral。\n"
        "5. drop_terms 是该条评论中应过滤的英文噪声词。\n"
        "6. 输出 JSON 必须符合这个结构：\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def clean_english_phrase(text: str) -> str:
    text = re.sub(r"[^A-Za-z'\s-]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def phrase_is_in_comment(phrase: str, comment: str) -> bool:
    phrase = clean_english_phrase(phrase)
    comment_clean = clean_english_phrase(comment)
    if not phrase or not comment_clean:
        return False
    if phrase in comment_clean:
        return True
    phrase_tokens = set(ta.tokenize_en(phrase))
    comment_tokens = set(ta.tokenize_en(comment_clean))
    return bool(phrase_tokens) and phrase_tokens.issubset(comment_tokens)


def tokenize_en_viewpoint_phrase(phrase: str) -> list[str]:
    clean_phrase = clean_english_phrase(phrase)
    raw_words = clean_phrase.split()
    tokens = ta.tokenize_en(clean_phrase)
    if not tokens:
        return []
    if len(tokens) <= 4 and any(word in EN_NEGATION_WORDS for word in raw_words):
        return [f"not_{'_'.join(tokens)}"]
    if 1 < len(tokens) <= 4:
        return ["_".join(tokens)]
    return tokens


def normalize_alignment_rows(raw_rows: list[dict[str, Any]], comments: list[str]) -> list[dict[str, Any]]:
    by_index = {int(row.get("comment_index", -1)): row for row in raw_rows if isinstance(row, dict)}
    normalized_rows: list[dict[str, Any]] = []
    for index, comment in enumerate(comments):
        raw = by_index.get(index, {})
        alignments = raw.get("alignments", []) if isinstance(raw, dict) else []
        en_terms = normalize_word_list(raw.get("en_viewpoint_terms")) if isinstance(raw, dict) else []
        zh_terms = normalize_word_list(raw.get("zh_viewpoint_terms")) if isinstance(raw, dict) else []

        accepted_en_tokens: list[str] = []
        accepted_zh_tokens: list[str] = []
        accepted_alignments: list[dict[str, Any]] = []

        for term in en_terms:
            if phrase_is_in_comment(term, comment):
                accepted_en_tokens.extend(tokenize_en_viewpoint_phrase(term))

        if isinstance(alignments, list):
            for item in alignments:
                if not isinstance(item, dict):
                    continue
                zh = str(item.get("zh", "")).strip()
                en = str(item.get("en", "")).strip()
                if not zh or not en or not phrase_is_in_comment(en, comment):
                    continue
                en_tokens = tokenize_en_viewpoint_phrase(en)
                zh_tokens = ta.tokenize_zh(zh)
                if not en_tokens or not zh_tokens:
                    continue
                accepted_en_tokens.extend(en_tokens)
                accepted_zh_tokens.extend(zh_tokens)
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                polarity = str(item.get("polarity", "neutral")).strip().lower()
                if polarity not in {"positive", "negative", "neutral"}:
                    polarity = "neutral"
                accepted_alignments.append(
                    {
                        "zh": " ".join(zh_tokens),
                        "en": " ".join(en_tokens),
                        "dimension": str(item.get("dimension", "")).strip(),
                        "polarity": polarity,
                        "confidence": max(0.0, min(confidence, 1.0)),
                    }
                )

        for zh_term in zh_terms:
            accepted_zh_tokens.extend(ta.tokenize_zh(zh_term))

        normalized_rows.append(
            {
                "comment_index": index,
                "en_pivot_zh_viewpoint_tokens": list(dict.fromkeys(accepted_zh_tokens)),
                "en_aligned_viewpoint_tokens": list(dict.fromkeys(accepted_en_tokens)),
                "en_viewpoint_alignments": accepted_alignments,
                "drop_terms": normalize_word_list(raw.get("drop_terms")) if isinstance(raw, dict) else [],
            }
        )
    return normalized_rows


def extract_english_viewpoints_with_deepseek(
    comments: list[str],
    video_meta: dict[str, Any],
    strategy: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    cache_path: Path | None = None,
    force_refresh: bool = False,
    batch_size: int = 30,
) -> list[dict[str, Any]]:
    cache_key = build_english_viewpoint_cache_key(video_meta, strategy, comments)
    if cache_path and not force_refresh:
        cached = read_cached_english_viewpoints(cache_path, cache_key)
        if cached is not None:
            return cached

    if not api_key or not comments:
        return []

    results: list[dict[str, Any]] = [
        {
            "comment_index": index,
            "en_pivot_zh_viewpoint_tokens": [],
            "en_aligned_viewpoint_tokens": [],
            "en_viewpoint_alignments": [],
            "drop_terms": [],
        }
        for index in range(len(comments))
    ]

    indexed_comments = [
        {"comment_index": index, "text": str(comment)[:360]}
        for index, comment in enumerate(comments)
        if ta.split_language_text(str(comment))[1].strip()
    ]
    batches = chunked(indexed_comments, batch_size)
    success_count = 0
    for batch in batches:
        local_comments = [str(item["text"]) for item in batch]
        reindexed_batch = [
            {"comment_index": local_index, "text": item["text"]}
            for local_index, item in enumerate(batch)
        ]
        messages = english_viewpoint_prompt(video_meta, strategy, reindexed_batch)
        try:
            payload = deepseek_chat_json(api_key, messages, model=model, temperature=0.1, timeout=150)
        except Exception:
            continue
        success_count += 1
        raw_rows = payload.get("comments", []) if isinstance(payload, dict) else []
        normalized = normalize_alignment_rows(raw_rows, local_comments)
        for local_row in normalized:
            original_index = int(batch[int(local_row["comment_index"])]["comment_index"])
            local_row["comment_index"] = original_index
            results[original_index] = local_row

    if cache_path and batches and success_count == len(batches):
        write_cached_english_viewpoints(cache_path, cache_key, results)
    return results


def apply_alias(token: str, aliases: dict[str, str]) -> str:
    return aliases.get(token.lower(), token)


def dedupe_tokens(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for token in tokens:
        text = str(token).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def expanded_viewpoint_terms(lang_strategy: dict[str, Any], language: Literal["zh", "en"]) -> set[str]:
    terms = set(normalize_word_list(lang_strategy.get("viewpoint_terms")))
    tokenizer = ta.tokenize_zh if language == "zh" else ta.tokenize_en
    expanded = set(terms)
    for term in terms:
        expanded.update(tokenizer(term))
    return {term for term in expanded if term}


def filter_tokens(
    tokens: list[str],
    lang_strategy: dict[str, Any],
    language: Literal["zh", "en"],
) -> tuple[list[str], list[str], list[str], list[str]]:
    drop_terms = set(normalize_word_list(lang_strategy.get("drop_terms")))
    anchor_terms = set(normalize_word_list(lang_strategy.get("anchor_terms")))
    viewpoint_terms = expanded_viewpoint_terms(lang_strategy, language)
    aliases = normalize_aliases(lang_strategy.get("aliases"))

    viewpoint_tokens: list[str] = []
    anchor_tokens: list[str] = []
    candidate_tokens: list[str] = []
    opinion_candidate_tokens: list[str] = []
    for raw_token in tokens:
        token = apply_alias(str(raw_token).strip().lower(), aliases)
        if not token or token in drop_terms:
            continue
        if token in anchor_terms:
            anchor_tokens.append(token)
            continue
        candidate_tokens.append(token)
        local_strategy = {"zh": lang_strategy, "en": lang_strategy}
        if is_opinion_like_token(token, language, local_strategy):
            opinion_candidate_tokens.append(token)
        if viewpoint_terms:
            if token in viewpoint_terms:
                viewpoint_tokens.append(token)
        else:
            viewpoint_tokens.append(token)
    return viewpoint_tokens, anchor_tokens, candidate_tokens, opinion_candidate_tokens


def should_relax_viewpoint_filter(strict_total: int, candidate_total: int) -> bool:
    if candidate_total <= 0:
        return False
    return strict_total < 12 or (strict_total / candidate_total) < 0.15


def select_viewpoint_tokens(
    strict_tokens: list[str],
    opinion_candidate_tokens: list[str],
    relaxed: bool,
) -> tuple[list[str], str]:
    strict_tokens = dedupe_tokens(strict_tokens)
    opinion_candidate_tokens = dedupe_tokens(opinion_candidate_tokens)
    if not relaxed:
        return strict_tokens, "viewpoint_terms" if strict_tokens else "empty_strict"
    merged = dedupe_tokens(strict_tokens + opinion_candidate_tokens)
    if merged and strict_tokens and len(merged) > len(strict_tokens):
        return merged, "viewpoint_terms_plus_opinion"
    if merged and strict_tokens:
        return merged, "viewpoint_terms"
    if merged:
        return merged, "opinion_fallback"
    return [], "insufficient_viewpoint"


def apply_viewpoint_strategy(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    video_meta: dict[str, Any] | None = None,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    english_cache_path: Path | None = None,
    use_deepseek_english: bool = False,
    force_english_refresh: bool = False,
) -> pd.DataFrame:
    texts = df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    normalized = enhance_strategy_for_texts(strategy, texts)
    enriched = df.copy()
    rows: list[dict[str, Any]] = []
    staged_rows: list[dict[str, Any]] = []
    for text in texts:
        zh_tokens = ta.tokenize_zh(text)
        en_tokens = ta.tokenize_en(text)
        zh_viewpoints, zh_anchors, zh_candidates, zh_opinion_candidates = filter_tokens(zh_tokens, normalized["zh"], "zh")
        en_viewpoints, en_anchors, en_candidates, en_opinion_candidates = filter_tokens(en_tokens, normalized["en"], "en")
        staged_rows.append(
            {
                "zh_viewpoints": zh_viewpoints,
                "en_viewpoints": en_viewpoints,
                "zh_candidates": zh_candidates,
                "en_candidates": en_candidates,
                "zh_opinion_candidates": zh_opinion_candidates,
                "en_opinion_candidates": en_opinion_candidates,
                "zh_anchors": zh_anchors,
                "en_anchors": en_anchors,
            }
        )

    zh_strict_total = sum(len(row["zh_viewpoints"]) for row in staged_rows)
    en_strict_total = sum(len(row["en_viewpoints"]) for row in staged_rows)
    zh_candidate_total = sum(len(row["zh_candidates"]) for row in staged_rows)
    en_candidate_total = sum(len(row["en_candidates"]) for row in staged_rows)
    zh_relaxed = should_relax_viewpoint_filter(zh_strict_total, zh_candidate_total)
    en_relaxed = should_relax_viewpoint_filter(en_strict_total, en_candidate_total)
    deepseek_en_rows: list[dict[str, Any]] = []
    if use_deepseek_english:
        try:
            deepseek_en_rows = extract_english_viewpoints_with_deepseek(
                texts,
                video_meta or {},
                normalized,
                api_key=api_key,
                model=model,
                cache_path=english_cache_path,
                force_refresh=force_english_refresh,
            )
        except Exception:
            deepseek_en_rows = []

    for index, staged in enumerate(staged_rows):
        if staged["zh_candidates"] or staged["zh_viewpoints"] or staged["zh_opinion_candidates"]:
            zh_viewpoints, zh_level = select_viewpoint_tokens(
                staged["zh_viewpoints"],
                staged["zh_opinion_candidates"],
                zh_relaxed,
            )
        else:
            zh_viewpoints, zh_level = [], "no_chinese"
        if staged["en_candidates"] or staged["en_viewpoints"] or staged["en_opinion_candidates"]:
            local_en_viewpoints, en_level = select_viewpoint_tokens(
                staged["en_viewpoints"],
                staged["en_opinion_candidates"],
                en_relaxed,
            )
        else:
            local_en_viewpoints, en_level = [], "no_english"
        en_result = deepseek_en_rows[index] if index < len(deepseek_en_rows) else {}
        aligned_en_tokens = [
            str(token).strip().lower()
            for token in en_result.get("en_aligned_viewpoint_tokens", [])
            if str(token).strip()
        ] if isinstance(en_result, dict) else []
        aligned_en_tokens = [
            token for token in dedupe_tokens(aligned_en_tokens)
            if is_opinion_like_token(token, "en", normalized)
        ]
        pivot_zh_tokens = [
            str(token).strip()
            for token in en_result.get("en_pivot_zh_viewpoint_tokens", [])
            if str(token).strip()
        ] if isinstance(en_result, dict) else []
        alignments = en_result.get("en_viewpoint_alignments", []) if isinstance(en_result, dict) else []
        if aligned_en_tokens:
            en_viewpoints = dedupe_tokens(aligned_en_tokens)
            en_mode = "deepseek_aligned"
            en_extraction_mode = "deepseek_aligned"
            en_level = "deepseek_aligned"
        else:
            en_viewpoints = local_en_viewpoints
            en_mode = en_level
            if en_viewpoints:
                en_extraction_mode = "local_fallback"
            elif en_level == "no_english":
                en_extraction_mode = "no_english"
            else:
                en_extraction_mode = "no_aligned_viewpoint"
        zh_anchors = staged["zh_anchors"]
        en_anchors = staged["en_anchors"]
        viewpoint_tokens = zh_viewpoints + en_viewpoints
        anchor_tokens = zh_anchors + en_anchors
        rows.append(
            {
                "zh_viewpoint_text": " ".join(zh_viewpoints),
                "en_viewpoint_text": " ".join(en_viewpoints),
                "viewpoint_text": " ".join(viewpoint_tokens),
                "zh_viewpoint_tokens": zh_viewpoints,
                "en_viewpoint_tokens": en_viewpoints,
                "viewpoint_tokens": viewpoint_tokens,
                "zh_anchor_tokens": zh_anchors,
                "en_anchor_tokens": en_anchors,
                "anchor_tokens": anchor_tokens,
                "zh_viewpoint_filter_mode": zh_level,
                "en_viewpoint_filter_mode": en_mode,
                "zh_viewpoint_fallback_level": zh_level,
                "en_viewpoint_fallback_level": en_level,
                "en_pivot_zh_viewpoint_tokens": pivot_zh_tokens,
                "en_aligned_viewpoint_tokens": aligned_en_tokens,
                "en_viewpoint_alignments": alignments if isinstance(alignments, list) else [],
                "en_viewpoint_extraction_mode": en_extraction_mode,
            }
        )

    viewpoint_df = pd.DataFrame(rows, index=enriched.index)
    for column in viewpoint_df.columns:
        enriched[column] = viewpoint_df[column]
    return enriched


def viewpoint_docs_and_tokens(viewpoint_df: pd.DataFrame, language: str) -> tuple[list[str], list[list[str]], list[str]]:
    if language == "中文":
        column = "zh_viewpoint_text"
    elif language == "英文":
        column = "en_viewpoint_text"
    else:
        column = "viewpoint_text"
    docs = viewpoint_df[column].fillna("").astype(str).tolist()
    token_lists = [doc.split() for doc in docs if doc.strip()]
    tokens = [token for items in token_lists for token in items]
    return docs, token_lists, tokens


def tokens_from_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [token.strip().lower() for token in str(value or "").split() if token.strip()]


def row_language_tokens(row: pd.Series, language: str) -> tuple[list[str], list[str]]:
    zh_tokens = tokens_from_value(row.get("zh_viewpoint_tokens", [])) or tokens_from_value(row.get("zh_viewpoint_text", ""))
    en_tokens = tokens_from_value(row.get("en_viewpoint_tokens", [])) or tokens_from_value(row.get("en_viewpoint_text", ""))
    if language == "中文":
        return zh_tokens, []
    if language == "英文":
        return [], en_tokens
    return zh_tokens, en_tokens


def purify_language_tokens(
    tokens: list[str],
    language: Literal["zh", "en"],
    strategy: dict[str, Any],
) -> list[str]:
    purified: list[str] = []
    for token in tokens:
        text = str(token).strip().lower()
        if not text:
            continue
        if is_opinion_like_token(text, language, strategy):
            purified.append(text)
    return purified


def purified_row_tokens(row: pd.Series, language: str, strategy: dict[str, Any]) -> list[str]:
    zh_tokens, en_tokens = row_language_tokens(row, language)
    return purify_language_tokens(zh_tokens, "zh", strategy) + purify_language_tokens(en_tokens, "en", strategy)


def purified_viewpoint_docs_and_tokens(
    viewpoint_df: pd.DataFrame,
    language: str,
    strategy: dict[str, Any],
) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    texts = viewpoint_df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    normalized = enhance_strategy_for_texts(strategy, texts)
    docs: list[str] = []
    source_comments: list[str] = []
    token_lists: list[list[str]] = []
    tokens: list[str] = []
    for _, row in viewpoint_df.iterrows():
        row_tokens = purified_row_tokens(row, language, normalized)
        if not row_tokens:
            continue
        token_lists.append(row_tokens)
        tokens.extend(row_tokens)
        docs.append(" ".join(row_tokens))
        source_comments.append(str(row.get("text", "")))
    return docs, source_comments, token_lists, tokens


def viewpoint_rows_with_purified_tokens(
    viewpoint_df: pd.DataFrame,
    language: str,
    strategy: dict[str, Any],
) -> pd.DataFrame:
    texts = viewpoint_df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    normalized = enhance_strategy_for_texts(strategy, texts)
    keep_indexes = [
        index
        for index, row in viewpoint_df.iterrows()
        if purified_row_tokens(row, language, normalized)
    ]
    return viewpoint_df.loc[keep_indexes].copy()


def match_dimension(value: str, dimensions: list[str]) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text_lower = text.lower()
    for dimension in dimensions:
        dim_lower = dimension.lower()
        if text_lower == dim_lower or text_lower in dim_lower or dim_lower in text_lower:
            return dimension
    return text


def build_viewpoint_cooccurrence_network(
    viewpoint_df: pd.DataFrame,
    strategy: dict[str, Any],
    language: str = "全部",
    top_n: int = 80,
    min_edge_weight: int = 1,
) -> dict[str, Any]:
    """Build a co-occurrence graph from purified viewpoint tokens and LLM dimensions."""
    texts = viewpoint_df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    normalized = enhance_strategy_for_texts(strategy, texts)
    dimensions = [str(item).strip() for item in normalized.get("viewpoint_dimensions", []) if str(item).strip()]

    row_tokens: list[list[str]] = [
        purified_row_tokens(row, language, normalized)
        for _, row in viewpoint_df.iterrows()
    ]
    token_counts = Counter(token for tokens in row_tokens for token in tokens)
    vocabulary = {word for word, _ in token_counts.most_common(top_n)}
    graph_nodes: dict[str, dict[str, Any]] = {}
    graph_edges: Counter[tuple[str, str]] = Counter()

    def add_node(node_id: str, label: str, group: str, size: int = 1, title: str = "") -> None:
        if node_id not in graph_nodes:
            graph_nodes[node_id] = {"id": node_id, "label": label, "group": group, "size": size, "title": title}
        else:
            graph_nodes[node_id]["size"] = int(graph_nodes[node_id].get("size", 1)) + size

    for dimension in dimensions:
        add_node(f"dimension:{dimension}", dimension, "观点维度", size=14, title="DeepSeek 生成的观点维度")

    for token, count in token_counts.most_common(top_n):
        add_node(f"token:{token}", token.replace("_", " "), "观点词", size=int(count), title=f"观点词频: {count}")

    for row_index, (_, row) in enumerate(viewpoint_df.iterrows()):
        tokens = sorted({token for token in row_tokens[row_index] if token in vocabulary})
        if not tokens:
            continue
        node_ids = [f"token:{token}" for token in tokens]
        alignment_dimensions: set[str] = set()
        if language in {"全部", "英文"}:
            alignments = row.get("en_viewpoint_alignments", [])
            if isinstance(alignments, list):
                for item in alignments:
                    if not isinstance(item, dict):
                        continue
                    matched = match_dimension(item.get("dimension", ""), dimensions)
                    if matched:
                        alignment_dimensions.add(matched)
        for dimension in alignment_dimensions:
            dim_id = f"dimension:{dimension}"
            add_node(dim_id, dimension, "观点维度", size=4, title="DeepSeek 英文观点对齐维度")
            for token in tokens:
                graph_edges[tuple(sorted((dim_id, f"token:{token}")))] += 2
            node_ids.append(dim_id)
        for left_index, source in enumerate(sorted(set(node_ids))):
            for target in sorted(set(node_ids))[left_index + 1 :]:
                graph_edges[(source, target)] += 1

    edges = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in graph_edges.items()
        if source in graph_nodes and target in graph_nodes and weight >= min_edge_weight
    ]
    nodes = list(graph_nodes.values())
    return {"nodes": nodes, "edges": edges}


def topic_summary_records(topic_df: pd.DataFrame, topic_names: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    if topic_df.empty or "topic_id" not in topic_df.columns:
        return []
    rows = []
    for topic_id, group in topic_df.groupby("topic_id", sort=False):
        name_info = topic_names.get(str(topic_id), {})
        words = group.sort_values("rank")["word"].astype(str).head(8).tolist()
        rows.append(
            {
                "topic_id": str(topic_id),
                "name": str(name_info.get("name", topic_id)),
                "summary": str(name_info.get("summary", "")),
                "dimensions": "、".join(normalize_word_list(name_info.get("dimensions"))),
                "top_words": words,
            }
        )
    return rows


def build_viewpoint_topic_network(
    viewpoint_df: pd.DataFrame,
    topic_records: list[dict[str, Any]],
    topic_names: dict[str, dict[str, str]],
    strategy: dict[str, Any],
    language: str = "全部",
) -> dict[str, Any]:
    """Build a graph over viewpoint themes/dimensions, not raw noun co-occurrence."""
    graph_nodes: dict[str, dict[str, Any]] = {}
    graph_edges: Counter[tuple[str, str]] = Counter()
    texts = viewpoint_df.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    normalized = enhance_strategy_for_texts(strategy, texts)
    dimensions = [str(item) for item in normalized.get("viewpoint_dimensions", []) if str(item).strip()]

    def add_node(node_id: str, label: str, group: str, size: int = 1, title: str = "") -> None:
        if node_id not in graph_nodes:
            graph_nodes[node_id] = {"id": node_id, "label": label, "group": group, "size": size, "title": title}
        else:
            graph_nodes[node_id]["size"] = int(graph_nodes[node_id].get("size", 1)) + size

    for dimension in dimensions:
        add_node(f"dimension:{dimension}", dimension, "观点维度", size=18, title="DeepSeek 生成的观点维度")

    topic_word_sets: dict[str, set[str]] = {}
    for record in topic_records:
        topic_id = str(record.get("topic_id", ""))
        info = topic_names.get(topic_id, {})
        name = str(info.get("name", topic_id))
        top_words = [str(word) for word in record.get("top_words", []) if str(word).strip()]
        topic_word_sets[topic_id] = set(top_words)
        title = "<br>".join(
            [
                f"主题: {name}",
                f"观点词: {'、'.join(top_words[:8])}",
                f"解释: {info.get('summary', '')}",
            ]
        )
        add_node(f"topic:{topic_id}", name, "观点主题", size=24, title=title)
        assigned_dimensions = normalize_word_list(info.get("dimensions"))
        for assigned in assigned_dimensions:
            matched_dimension = next(
                (
                    dimension
                    for dimension in dimensions
                    if assigned == dimension.lower()
                    or assigned in dimension.lower()
                    or dimension.lower() in assigned
                ),
                None,
            )
            if matched_dimension:
                graph_edges[(f"topic:{topic_id}", f"dimension:{matched_dimension}")] += 4
        for dimension in dimensions:
            dim_key = dimension.lower()
            if any(dim_key in word.lower() or word.lower() in dim_key for word in top_words):
                graph_edges[(f"topic:{topic_id}", f"dimension:{dimension}")] += 2

    for _, row in viewpoint_df.iterrows():
        tokens = set(purified_row_tokens(row, language, normalized))
        matched_topics = [
            f"topic:{topic_id}"
            for topic_id, words in topic_word_sets.items()
            if words and tokens.intersection(words)
        ]
        for left_index, source in enumerate(matched_topics):
            for target in matched_topics[left_index + 1 :]:
                graph_edges[tuple(sorted((source, target)))] += 1

    nodes = list(graph_nodes.values())
    edges = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in graph_edges.items()
        if source in graph_nodes and target in graph_nodes and weight > 0
    ]
    return {"nodes": nodes, "edges": edges}


def analysis_to_jsonable(
    strategy: dict[str, Any],
    viewpoint_df: pd.DataFrame,
    topic_df: pd.DataFrame | None,
    topic_names: dict[str, dict[str, str]] | None,
    topic_records: list[dict[str, Any]] | None,
    language: str,
) -> dict[str, Any]:
    _, _, token_lists, tokens = purified_viewpoint_docs_and_tokens(viewpoint_df, language, strategy)
    topic_df = topic_df if topic_df is not None else pd.DataFrame()
    topic_names = topic_names or {}
    topic_records = topic_records or []
    cooccurrence_network = build_viewpoint_cooccurrence_network(viewpoint_df, strategy, language)
    topic_network = (
        build_viewpoint_topic_network(viewpoint_df, topic_records, topic_names, strategy, language)
        if topic_records
        else cooccurrence_network
    )
    return {
        "strategy": normalize_strategy(strategy),
        "language": language,
        "viewpoint_word_counts": Counter(tokens).most_common(80),
        "viewpoint_comment_count": int(len(token_lists)),
        "topic_words": topic_df.to_dict(orient="records") if not topic_df.empty else [],
        "topic_summaries": topic_summary_records(topic_df, topic_names) if not topic_df.empty else [],
        "topic_network": topic_network,
        "viewpoint_cooccurrence_network": cooccurrence_network,
        "token_lists": token_lists[:300],
    }


def has_viewpoint_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in STRATEGY_COLUMNS)


def topic_naming_prompt(
    video_meta: dict[str, Any],
    strategy: dict[str, Any],
    topic_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "video": {
            "title": str(video_meta.get("title", "")),
            "description": str(video_meta.get("description", ""))[:1500],
        },
        "strategy": normalize_strategy(strategy),
        "topics": topic_records,
    }
    system = (
        "你是舆情主题命名助手。根据视频语境、观点策略、主题高权重词和代表评论，"
        "给每个主题起一个简短、业务可读的中文名称，并从策略给出的观点维度中选择最相关的维度。"
        "不要使用 Topic 1/2/3。只返回 JSON。"
    )
    user = (
        "请为每个主题命名，并给出一句解释。输出结构："
        '{"topics":[{"topic_id":"T1","name":"简短主题名","summary":"一句解释","dimensions":["观点维度"]}]}。'
        "dimensions 必须优先从输入 strategy.viewpoint_dimensions 中选择，可以为空数组但不要自造固定分类。"
        f"\n输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fallback_topic_names(topic_records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    names: dict[str, dict[str, str]] = {}
    for record in topic_records:
        topic_id = str(record.get("topic_id", ""))
        words = [str(word) for word in record.get("top_words", []) if str(word).strip()]
        name = " / ".join(words[:3]) if words else topic_id
        names[topic_id] = {"name": f"观点：{name}", "summary": "基于观点词权重自动生成。", "dimensions": []}
    return names


def name_topics_with_deepseek(
    topic_records: list[dict[str, Any]],
    video_meta: dict[str, Any],
    strategy: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, dict[str, str]]:
    fallback = fallback_topic_names(topic_records)
    if not api_key or not topic_records:
        return fallback
    try:
        messages = topic_naming_prompt(video_meta, strategy, topic_records)
        payload = deepseek_chat_json(api_key, messages, model=model, temperature=0.1, timeout=150)
    except Exception:
        return fallback

    topics = payload.get("topics", []) if isinstance(payload, dict) else []
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic_id = str(item.get("topic_id", "")).strip()
        name = str(item.get("name", "")).strip()
        summary = str(item.get("summary", "")).strip()
        dimensions = normalize_word_list(item.get("dimensions"))
        if topic_id and name:
            fallback[topic_id] = {"name": name[:30], "summary": summary[:120], "dimensions": dimensions[:5]}
    return fallback
