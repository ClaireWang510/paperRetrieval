<script setup>
import axios from 'axios'
import { computed, ref } from 'vue'
import PaperCard from './components/PaperCard.vue'

const query = ref('')
const DEFAULT_TOP_K = 20

const loading = ref(false)
const error = ref('')
const answer = ref('')
const answerReferences = ref([])
const answerLoading = ref(false)
const answerError = ref('')
const results = ref([])
const currentPage = ref(1)
const executedQuery = ref('')

const PAGE_SIZE = 10

const hasResults = computed(() => results.value.length > 0)
const totalPages = computed(() => Math.max(1, Math.ceil(results.value.length / PAGE_SIZE)))
const paginatedResults = computed(() => {
    const start = (currentPage.value - 1) * PAGE_SIZE
    return results.value.slice(start, start + PAGE_SIZE)
})
const pageStart = computed(() => (results.value.length ? (currentPage.value - 1) * PAGE_SIZE + 1 : 0))
const pageEnd = computed(() => Math.min(currentPage.value * PAGE_SIZE, results.value.length))

const doSearch = async () => {
    if (!query.value.trim()) {
        error.value = '请输入检索内容。'
        return
    }

    loading.value = true
    error.value = ''
    answer.value = ''
    answerReferences.value = []
    answerLoading.value = false
    answerError.value = ''
    results.value = []
    currentPage.value = 1
    executedQuery.value = ''

    try {
        const resp = await axios.post('/api/search', {
            query: query.value.trim(),
            top_k: DEFAULT_TOP_K,
        })
        const data = resp.data || {}
        executedQuery.value = query.value.trim()
        answer.value = ''
        answerReferences.value = []
        results.value = Array.isArray(data.results) ? data.results : []
    } catch (e) {
        error.value = e?.response?.data?.error || e?.message || '请求失败'
    } finally {
        loading.value = false
    }
}

const goToPrevPage = () => {
    if (currentPage.value > 1) currentPage.value -= 1
}

const goToNextPage = () => {
    if (currentPage.value < totalPages.value) currentPage.value += 1
}

const generateAnswer = async () => {
    if (!executedQuery.value.trim() || answerLoading.value) return
    answerLoading.value = true
    answerError.value = ''
    answer.value = ''
    answerReferences.value = []
    try {
        const resp = await axios.post('/api/answer', {
            query: executedQuery.value.trim(),
            top_n_papers: 10,
            top_n_units: 3,
        })
        const data = resp?.data || {}
        answer.value = data.answer || ''
        answerReferences.value = Array.isArray(data.answer_references) ? data.answer_references : []
    } catch (e) {
        answerError.value = e?.response?.data?.error || e?.message || '回答生成失败'
    } finally {
        answerLoading.value = false
    }
}
</script>

<template>
    <div class="page-shell">
        <main class="page">
        <section class="hero">
            <p class="tag">Scientific Resource Assistant</p>
            <h1>智能科学研究助手</h1>
            <p class="sub">让问题、证据与论文来源在一个界面里清晰连接。</p>
        </section>

        <section class="panel search-panel">
            <label class="label" for="query">输入检索目标</label>
            <textarea id="query" v-model="query" rows="4" placeholder="例如：近年来用 Transformer 做时序预测的方法，以及其在金融数据上的评估" />

            <button class="search-btn" :disabled="loading" @click="doSearch">
                {{ loading ? '检索中...' : '开始检索' }}
            </button>

            <p v-if="error" class="error">{{ error }}</p>
        </section>

        <section v-if="hasResults" class="panel">
            <div class="answer-toolbar">
                <h2>基于语义证据的简要回答</h2>
                <button class="search-btn" :disabled="answerLoading" @click="generateAnswer">
                    {{ answerLoading ? '生成中...' : '生成简要回答' }}
                </button>
            </div>
            <p v-if="answerError" class="error">{{ answerError }}</p>
            <p v-if="answer" class="answer">{{ answer }}</p>
            <p v-else-if="!answerLoading" class="muted">当前未生成回答，点击上方按钮按需生成。</p>
            <div v-if="answerReferences.length > 0" class="references">
                <p class="references-title">引用文章</p>
                <ul>
                    <li v-for="ref in answerReferences" :key="`${ref.paper_index}-${ref.arxiv_id}`">
                        [{{ ref.paper_index }}] {{ ref.title || '无标题' }} (arXiv: {{ ref.arxiv_id || '-' }})
                    </li>
                </ul>
            </div>
        </section>

        <section v-if="hasResults" class="results">
            <div class="results-toolbar">
                <p class="results-summary">第 {{ pageStart }}-{{ pageEnd }} 篇，共 {{ results.length }} 篇</p>
                <div v-if="totalPages > 1" class="pagination">
                    <button class="page-btn" :disabled="currentPage === 1" @click="goToPrevPage">上一页</button>
                    <span class="page-indicator">{{ currentPage }} / {{ totalPages }}</span>
                    <button class="page-btn" :disabled="currentPage === totalPages" @click="goToNextPage">下一页</button>
                </div>
            </div>

            <PaperCard
                v-for="item in paginatedResults"
                :key="item.arxiv_id || item.title"
                :item="item"
                :search-query="executedQuery"
            />
        </section>
        </main>

        <!--
        <section class="agent-page-wrap">
            <AgentWorkspace />
        </section>
        -->
    </div>
</template>

<style scoped>
:global(body) {
    margin: 0;
    background:
        radial-gradient(circle at 10% 8%, rgba(232, 147, 45, 0.19), transparent 34%),
        radial-gradient(circle at 92% 3%, rgba(21, 94, 113, 0.17), transparent 31%),
        linear-gradient(155deg, #f7f3ea 0%, #edf5f7 48%, #fffdf7 100%);
    font-family: 'Source Han Sans SC', 'Noto Sans SC', 'PingFang SC', sans-serif;
    color: #1a2b2f;
}

.page {
    max-width: 1120px;
    margin: 0 auto;
    padding: 2.2rem 1rem 3rem;
}

.page-shell {
    min-height: 100vh;
}

.hero {
    margin-bottom: 1.35rem;
    animation: fadeUp 440ms ease-out;
}

.tag {
    display: inline-block;
    padding: 0.2rem 0.64rem;
    border: 1px solid #0f6270;
    border-radius: 999px;
    color: #0f6270;
    font-size: 0.76rem;
    letter-spacing: 0.07em;
    font-weight: 700;
}

h1 {
    margin: 0.55rem 0 0.42rem;
    font-size: clamp(2rem, 4.2vw, 3.1rem);
    font-family: 'Source Han Serif SC', 'Noto Serif SC', serif;
    color: #093a45;
    letter-spacing: 0.02em;
}

.sub {
    margin: 0;
    color: #38565d;
    font-size: 1.01rem;
}

.panel {
    background: rgba(255, 255, 255, 0.84);
    backdrop-filter: blur(3px);
    border: 1px solid rgba(15, 82, 95, 0.15);
    border-radius: 18px;
    padding: 1.08rem;
    margin-bottom: 1.05rem;
    animation: fadeUp 520ms ease-out;
}

.search-panel {
    animation-delay: 90ms;
}

.label {
    font-size: 0.86rem;
    display: inline-block;
    margin-bottom: 0.35rem;
    color: #22424a;
}

textarea,
input,
select {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #b7cfd5;
    border-radius: 12px;
    padding: 0.7rem 0.76rem;
    font-size: 0.96rem;
    background: #fff;
}

textarea:focus,
input:focus,
select:focus {
    outline: 2px solid #2f9dad;
    border-color: #2f9dad;
}

.search-btn {
    margin-top: 0.95rem;
    border: none;
    background: linear-gradient(102deg, #0f5a66, #2a7a8e);
    color: #fff;
    padding: 0.68rem 1.18rem;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 700;
}

.search-btn:disabled {
    opacity: 0.7;
    cursor: wait;
}

.error {
    margin-top: 0.6rem;
    color: #a11b1b;
}

h2 {
    margin: 0 0 0.4rem;
    color: #0f4e58;
}

.answer-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 0.4rem;
}

.answer-toolbar .search-btn {
    margin-top: 0;
}

.answer {
    margin: 0;
    line-height: 1.65;
}

.muted {
    color: #4e6b72;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

.references {
    margin-top: 0.9rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(15, 89, 100, 0.15);
}

.references-title {
    margin: 0 0 0.4rem;
    color: #0f4e58;
    font-weight: 700;
}

.results {
    display: grid;
    gap: 0.9rem;
}

.results-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.results-summary {
    margin: 0;
    color: #35535a;
    font-weight: 600;
}

.pagination {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid #d3e3e6;
    border-radius: 999px;
    padding: 0.28rem 0.34rem;
}

.page-btn {
    border: 1px solid #8fb6bd;
    background: rgba(255, 255, 255, 0.88);
    color: #134c55;
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    cursor: pointer;
}

.page-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

.page-indicator {
    color: #35535a;
    font-size: 0.92rem;
    min-width: 3.7rem;
    text-align: center;
}

ul {
    padding-left: 1.1rem;
}

li {
    margin: 0.35rem 0;
    color: #2e4e55;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 840px) {
    .results-toolbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .page {
        padding-top: 1.4rem;
    }

    .top-nav-wrap {
        padding-top: 0.8rem;
    }

    .pagination {
        width: 100%;
        justify-content: space-between;
    }
}
</style>
