# YouTube 评论舆情分析工具

这是一个基于 Streamlit 的 YouTube 评论舆情分析工具。项目支持评论抓取、上传表格分析、中英文分开处理、情感分析、词云、语义网络、传统 LDA/NMF 主题模型，以及基于 DeepSeek 的动态观点分析。

## 现在的核心逻辑

项目目前有两条分析链路，主页面“分析”区域已拆成两个大模块：

1. **传统文本分析链路**
   - YouTube Data API 抓取评论，或上传 CSV / Excel。
   - 自动识别评论语言，把中文和英文分开处理。
   - 中文使用 `jieba` 分词、`SnowNLP` 情感分析、中文字体词云。
   - 英文使用英文 tokenizer、NLTK 停用词、补充停用词、词形还原、VADER 情感分析、英文词云。
   - 情感结果统一输出为 `正面 / 中性 / 负面`。
   - 保留中英文分开的词频、词云、语义共现网络。
   - 额外支持双语主题合并：英文核心词会先映射或免费翻译成中文主题，再和中文词频合并展示。

2. **DeepSeek 动态观点分析链路**
   - 系统先拿到视频上下文：视频标题、视频描述、频道信息、标签。
   - DeepSeek 根据视频上下文和评论样本动态判断：
     - 视频类型
     - 视频核心主题
     - 评论里真正值得看的观点维度
   - DeepSeek 动态生成该视频专属清洗策略：
     - `drop_terms`：本视频语境下无意义的高频词、主体名词、地名、人名、频道名等
     - `viewpoint_terms`：观点、评价、情感、体验、建议、争议表达
     - `anchor_terms`：可解释语境但不应该主导词频和网络的主题锚点词
     - `aliases`：同义观点词归一
   - 本地代码再把策略应用到评论：
     - 只保留观点类文本
     - 剔除主体名词和无意义名词
     - 中英文分别处理
     - 在 DeepSeek 策略基础上自动补充动态 `drop_terms`，过滤代词、连接词、平台词、人物名、地名和高频泛名词。
     - 如果 `viewpoint_terms` 匹配过少，只补充评价、情绪、体验、建议类观点候选词，不再退回原始高频词。
     - 英文评论会额外通过 DeepSeek 抽取“中文观点词 - 英文原词”对齐结果，并记录观点维度和极性；只有出现在原评论中的英文词/短语才会进入英文观点词云和共现网络。
     - `not true` 这类否定短语会保留为 `not_true` 形式，避免被普通停用词拆散。
   - 清洗后的观点文本在进入 LDA/NMF 前还会做一次二次净化，确保主题模型只聚类观点词。
   - LDA/NMF 仍负责本地聚类，DeepSeek 负责给主题自动命名，所以结果不再显示 `Topic 1 / Topic 2`。
   - 观点关联网络不再用原始名词共现，而是优先展示“净化观点词共现”；运行 LDA/NMF 后再展示“自动命名观点主题 + DeepSeek 观点维度”网络。

两个模块下面都包含：

- LDA 主题模型
- NMF 主题模型
- networkx 语义网络
- 情感类别饼图
- 情感极性饼图
- 词云图
- 双语合并分析

区别在于：

- **传统分析** 使用本地中英文分词结果，适合保留原始词频和共现结构。
- **DeepSeek 大模型分析** 先做动态观点清洗，只让观点词进入主题模型、词云、网络和双语合并。

## 视频上下文从哪里来

抓取 YouTube 评论时，视频上下文会自动通过 YouTube Data API 获取，不需要手动输入。

自动调用的是 YouTube `videos` 接口：

```text
https://www.googleapis.com/youtube/v3/videos
part=snippet
```

会保存这些字段：

- `title`
- `description`
- `channel_title`
- `tags`
- `published_at`
- `video_id`

这些信息会写入：

```text
data/current_video_meta.json
```

侧边栏里的“手动补充/覆盖视频上下文”只用于两种情况：

- 上传历史 CSV / Excel，没有视频 ID，也没有视频标题和描述。
- YouTube API 没有成功获取标题/描述，但仍想让 DeepSeek 有上下文。

正常抓取视频时不需要手写标题和描述。

## API Key 配置

项目现在支持从 `.env` 自动读取 Key，不需要每次在页面里输入。

当前已创建 `.env` 文件：

```env
YOUTUBE_API_KEY=
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
```

把自己的 Key 填进去即可。页面侧边栏里的 Key 输入框只是临时覆盖 `.env` 的值。

变量说明：

- `YOUTUBE_API_KEY`：Google Cloud Console 里的 YouTube Data API v3 Key，用于抓评论和视频标题/描述。
- `DEEPSEEK_API_KEY`：DeepSeek API Key，用于动态清洗策略和主题命名。
- `DEEPSEEK_MODEL`：默认 `deepseek-chat`，也可以改为 `deepseek-reasoner`。

项目也提供了 `.env.example`，用于查看变量格式。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

如果 8501 端口被占用，可以指定端口：

```bash
streamlit run app.py --server.port 8502
```

## 使用流程

### 抓取新视频

1. 在 `.env` 中填好 `YOUTUBE_API_KEY` 和 `DEEPSEEK_API_KEY`。
2. 启动项目。
3. 在侧边栏输入视频链接或视频 ID。
4. 设置最大评论数。
5. 点击“抓取评论”。
6. 系统会同时抓取：
   - 评论数据
   - 视频标题
   - 视频描述
   - 频道名
   - 标签
7. 在“分析模块”选择 `DeepSeek 大模型分析`。
8. 点击“加载 / 生成观点策略”。
9. 在 LDA 或 NMF 页签中运行观点主题模型，得到自动命名后的观点主题。
10. 查看观点词云、观点网络、情感图、双语合并图。

### 上传历史表格

1. 上传包含 `text` 列的 CSV / Excel。
2. 如果没有视频元数据，展开“上传表格时手动补充/覆盖视频上下文”。
3. 填写视频标题和描述。
4. 再生成 DeepSeek 观点策略。

## 页面功能说明

### DeepSeek 大模型分析

这是现在更推荐看的分析入口。

它解决传统 LDA/NMF 和共现网络容易被名词主导的问题。DeepSeek 不直接替代本地聚类，而是先动态生成清洗策略，让后续分析只看观点词。

输出包括：

- 视频类型
- 核心主题
- 观点维度
- 动态清洗策略 JSON
- 观点词频
- 观点词云
- LDA 观点主题模型
- NMF 观点主题模型
- 观点主题-观点维度网络
- DeepSeek 自动命名的观点主题
- 观点情感类别饼图
- 观点情感极性饼图
- 观点双语合并词频图和词云

### 传统分析

保留原来的本地分析能力，用于对比或快速探索。

现在支持中文和英文分开建模，避免中英文混在一起导致分词和主题不准确。

传统分析包含：

- 中文 / 英文 LDA 主题模型
- 中文 / 英文 NMF 主题模型
- 中文 / 英文 networkx 共现网络
- 中英文情感分析
- 中文 / 英文词云
- 双语合并词频图和双语词云

### 语义网络

保留原来的中英文分开共现网络。

DeepSeek 大模型分析里的“观点关联网络”不是原始词共现网络。它现在分两层：

- 未运行 LDA/NMF 时，展示二次净化后的观点词共现网络。
- 运行 LDA/NMF 后，展示自动命名后的观点主题与 DeepSeek 观点维度之间的网络。

这样避免视频主体名词、地名、人名、频道名成为网络中心。

### 词云

包括三类：

- 中文词云
- 英文词云
- 双语主题合并词云

DeepSeek 观点分析页里还有“观点词云”，只展示观点词。

### 情感分析

统一输出：

- 正面
- 中性
- 负面

中文使用 SnowNLP，英文使用 VADER。

## 数据和缓存文件

项目会在 `data/` 下保存当前分析状态：

```text
data/current_comments.csv
data/current_video_meta.json
data/current_viewpoint_strategy.json
data/current_llm_analysis.json
data/current_llm_english_viewpoints.json
```

说明：

- `current_comments.csv`：当前抓取或上传的评论数据。
- `current_video_meta.json`：当前视频标题、描述、频道、标签等上下文。
- `current_viewpoint_strategy.json`：DeepSeek 生成的动态观点清洗策略。
- `current_llm_analysis.json`：主页面最后一次 DeepSeek LDA/NMF 观点主题分析结果，包括自动命名主题、主题词权重和观点主题网络。
- `current_llm_english_viewpoints.json`：DeepSeek 英文评论观点抽取缓存，保存每条英文评论的中文观点词、英文对应词、观点维度、极性和对齐关系。

大屏页面会优先读取主页面当前 session 中的数据；如果没有 session 数据，就读取这些缓存文件。

### 缓存策略

缓存分四层：

1. **视频上下文缓存**
   - 文件：`data/current_video_meta.json`
   - 来源：YouTube Data API `videos` 接口。
   - 作用：保存标题、描述、频道、标签、发布时间和视频 ID。
   - 刷新时机：重新抓取新视频，或在侧边栏手动覆盖视频上下文。

2. **DeepSeek 观点策略缓存**
   - 文件：`data/current_viewpoint_strategy.json`
   - 缓存 Key 由这些内容计算：
     - 视频标题
     - 视频描述前 2000 字
     - 高赞评论样本
     - 中文候选高频词
     - 英文候选高频词
   - 如果这些输入没变，点击“加载 / 生成观点策略”会直接读缓存，不重新请求 DeepSeek。
   - 勾选“重新生成观点策略”会跳过旧缓存，重新调用 DeepSeek。
   - 如果没有匹配缓存且没有 `DEEPSEEK_API_KEY`，系统会提示需要配置 Key。
   - 观点过滤逻辑本身在本地执行；即使复用旧策略缓存，也会重新叠加本地动态 `drop_terms` 和观点候选回退，不需要为了普通停用词问题重新调用 DeepSeek。

3. **DeepSeek 主题分析缓存**
   - 文件：`data/current_llm_analysis.json`
   - 缓存 Key 由这些内容计算：
     - 视频标题
     - 视频描述前 2000 字
     - 当前 DeepSeek 观点策略
     - 评论样本
   - 主页面运行 DeepSeek LDA 或 NMF 后会写入。
   - 大屏的 DeepSeek 模块会读取这个缓存，用于展示自动命名主题、主题词权重图和观点主题网络。
   - 如果还没有主题分析缓存，大屏会用当前净化后的观点词生成观点共现网络作为回退展示。
   - 当前筛选器会影响大屏里的观点词频、观点情感和观点双语合并；主题模型缓存代表主页面最后一次完整分析结果。

4. **DeepSeek 英文观点对齐缓存**
   - 文件：`data/current_llm_english_viewpoints.json`
   - 缓存 Key 由视频标题、视频描述、当前观点策略和英文评论文本计算。
   - 主页面生成/加载观点策略时，如果配置了 `DEEPSEEK_API_KEY`，会批量让 DeepSeek 抽取英文评论中的观点词，并记录对应的中文观点词、观点维度和极性。
   - 大屏只读取这个缓存，不主动消耗 DeepSeek；没有缓存时回退到本地英文观点过滤。
   - DeepSeek 返回的英文词必须能在原评论中匹配到，否则会被丢弃，避免模型幻觉词进入词云或网络。

## 大屏 Dashboard

大屏页面位于：

```text
pages/dashboard.py
```

大屏侧边栏现在有“大屏模块”选择：

- `传统分析`
- `DeepSeek 大模型分析`

两个模块都会保留顶部的总览指标、声量走势、情感类别和情感极性。

传统模块展示：

- 评论总量
- 点赞总量
- 作者数量
- 正面占比
- 声量走势
- 情感类别
- 情感极性
- 中英文分开词频
- 中英文分开语义网络
- 双语主题合并
- 高赞评论
- 作者互动榜

DeepSeek 大模型模块展示：

- DeepSeek 观点词频
- DeepSeek 自动命名观点主题
- DeepSeek 观点主题词权重图
- DeepSeek 观点主题-观点维度网络
- DeepSeek 观点情感类别饼图
- DeepSeek 观点情感极性饼图
- DeepSeek 观点双语合并图
- DeepSeek 观点双语词云

如果没有生成观点策略，大屏会提示当前未检测到策略，但不会影响传统模块。

## DeepSeek 观点策略如何工作

DeepSeek 输入内容不是完整评论全集，而是：

- 视频标题
- 视频描述
- 频道名
- 标签
- 高赞评论样本
- 当前中英文高频候选词

这样做是为了减少成本和请求体大小。

DeepSeek 输出固定 JSON 结构，但具体内容完全根据当前视频动态生成：

```json
{
  "video_type": "视频类型",
  "core_theme": "核心主题",
  "viewpoint_dimensions": ["观点维度"],
  "zh": {
    "drop_terms": [],
    "viewpoint_terms": [],
    "anchor_terms": [],
    "aliases": {}
  },
  "en": {
    "drop_terms": [],
    "viewpoint_terms": [],
    "anchor_terms": [],
    "aliases": {}
  }
}
```

代码不会写死固定业务分类，也不会写死固定观点类别。代码只负责：

- 定义 JSON 结构
- 调用 DeepSeek
- 解析和缓存策略
- 缓存英文评论的中文观点词与英文原词对齐结果
- 把策略应用到评论
- 做本地统计、聚类和可视化

## 当前限制

- DeepSeek 观点分析需要有效的 `DEEPSEEK_API_KEY`。
- 如果 DeepSeek 返回的 JSON 不合法、请求中断或返回不完整，系统会重试最多 3 次；策略生成还会自动缩短输入再试，传统分析仍可用。
- 如果上传历史表格但不提供标题/描述，DeepSeek 只能根据评论样本生成弱策略。
- NLTK 的 `stopwords` 和 `wordnet` 语料如果本地没有下载，英文处理会自动退回内置停用词和规则词形还原，不会中断运行。
