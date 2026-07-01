# Scientific Resource Prototype (Flask + Vue)

该原型实现了：

- 前端：Vue + Vite 检索界面
- 后端：Flask API
- 检索引擎：直接复用 `release/src/scientific_resource_release` 中的检索服务
- 数据库：直接连接已有 PostgreSQL 数据库 `scientific_resource`

## 目录结构

- `backend/app.py`: Flask API（`/api/health`、`/api/search`）
- `backend/requirements.txt`: 后端依赖
- `frontend/`: Vue 前端工程

## 1. 后端启动

在项目根目录 `ScientificResource` 下：

```bash
conda activate ScientificResource
cd release/prototype/backend

# 安装原型后端依赖
pip install -r requirements.txt

# 安装 release 核心依赖（检索/向量/重排/LLM）
cd ../../
pip install -e .
cd prototype/backend

# 可选：环境变量
cp .env.example .env
export DATABASE_URL="postgresql://localhost:5432/scientific_resource"

# 关键：让 Python 能导入 release/src 下的包
PYTHONPATH=../../src python app.py
```

默认监听：`http://127.0.0.1:8001`

健康检查：

```bash
curl http://127.0.0.1:8001/api/health
```

## 2. 前端启动

新开一个终端，在 `ScientificResource` 下执行：

```bash
conda activate ScientificResource
cd release/prototype/frontend

# 若缺依赖则安装
npm install

# 启动开发服务（已代理 /api 到 8001）
npm run dev
```

打开：`http://127.0.0.1:5173`

## 3. 生产构建（可选）

```bash
cd release/prototype/frontend
npm run build
npm run preview
```

## 4. API 示例

`POST /api/search`

```json
{
  "query": "近年来用 Transformer 做时序预测的方法",
  "top_k": 20,
  "sparse_weight": 0.5,
  "intent_decomposer": "llm"
}
```

返回包含：

- `rewritten_query`
- `results`（含 `final_score`、`rerank_score`、`recommendation_reason`、`semantic_units`）
- `answer`

## 5. 说明

- 若未配置 `LLM_API_KEY`，系统仍可检索；推荐理由与顶部回答会退化为规则 fallback 文本。
- 如数据库地址非本机，请设置 `DATABASE_URL`。
- 首次运行 reranker 可能下载模型，耗时取决于网络。
