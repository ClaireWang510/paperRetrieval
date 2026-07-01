<script setup>
import axios from 'axios'
import { computed, ref } from 'vue'
import EvidenceCard from './EvidenceCard.vue'

const props = defineProps({
    item: {
        type: Object,
        required: true,
    },
    searchQuery: {
        type: String,
        default: '',
    },
})

const ABSTRACT_PREVIEW_LEN = 280

const normalizeArxivId = (value) => {
    if (!value) return ''
    const cleaned = String(value).replace(/^arXiv:/i, '').trim()
    const versionIndex = cleaned.indexOf('v')
    return versionIndex > 0 ? cleaned.slice(0, versionIndex) : cleaned
}

const normalizedId = computed(() => normalizeArxivId(props.item.arxiv_id))
const pdfUrl = computed(() =>
    normalizedId.value ? `/api/download/pdf/${encodeURIComponent(normalizedId.value)}` : '',
)
const latexUrl = computed(() =>
    normalizedId.value ? `/api/download/latex/${encodeURIComponent(normalizedId.value)}` : '',
)
const extraSemanticUnits = ref([])
const semanticUnits = computed(() =>
    extraSemanticUnits.value.length
        ? extraSemanticUnits.value
        : Array.isArray(props.item.semantic_units)
          ? props.item.semantic_units
          : [],
)
const hasDownloads = computed(() => Boolean(pdfUrl.value || latexUrl.value))

const abstractText = computed(() => props.item.abstract || '无摘要')
const abstractExpanded = ref(false)
const abstractNeedCollapse = computed(() => abstractText.value.length > ABSTRACT_PREVIEW_LEN)
const displayAbstract = computed(() => {
    if (!abstractNeedCollapse.value || abstractExpanded.value) return abstractText.value
    return `${abstractText.value.slice(0, ABSTRACT_PREVIEW_LEN)}...`
})

const evidenceExpanded = ref(false)
const reasonLoading = ref(false)
const reasonError = ref('')
const localReason = ref('')
const recommendationReason = computed(() => localReason.value || props.item.recommendation_reason || '')
const canGenerateReason = computed(() => Boolean(normalizedId.value && props.searchQuery.trim()))

const generateReason = async () => {
    if (!canGenerateReason.value || reasonLoading.value) return
    reasonLoading.value = true
    reasonError.value = ''
    try {
        const resp = await axios.post('/api/reason', {
            query: props.searchQuery,
            arxiv_id: props.item.arxiv_id,
            title: props.item.title,
            abstract: props.item.abstract,
            top_n: 3,
        })
        const data = resp?.data || {}
        localReason.value = data.recommendation_reason || ''
        extraSemanticUnits.value = Array.isArray(data.semantic_units) ? data.semantic_units : []
    } catch (e) {
        reasonError.value = e?.response?.data?.error || e?.message || '推荐理由生成失败'
    } finally {
        reasonLoading.value = false
    }
}

const displayEvidenceUnits = computed(() => {
    return evidenceExpanded.value ? semanticUnits.value : []
})
</script>

<template>
    <article class="paper-card">
        <header class="paper-head">
            <h3>{{ item.title || '无标题' }}</h3>
            <p class="meta-line">
                <span class="meta-chip">arXiv: {{ item.arxiv_id || '-' }}</span>
                <span class="meta-chip">引用 {{ item.citation_count ?? '-' }}</span>
                <span class="meta-chip venue">{{ item.venue || 'Venue 未提供' }}</span>
                <span class="meta-chip ccf">CCF {{ item.ccf_tier || '未收录' }}</span>
            </p>
            <p v-if="hasDownloads" class="download-row">
                <a v-if="pdfUrl" class="download-link" :href="pdfUrl">下载 PDF</a>
                <a v-if="latexUrl" class="download-link" :href="latexUrl">下载 LaTeX 源码</a>
            </p>
        </header>

        <p v-if="recommendationReason" class="reason">
            <strong>推荐理由：</strong>{{ recommendationReason }}
        </p>
        <div v-else class="reason-action-row">
            <button
                type="button"
                class="toggle-btn"
                :disabled="reasonLoading || !canGenerateReason"
                @click="generateReason"
            >
                {{ reasonLoading ? '生成中...' : '生成推荐理由' }}
            </button>
            <p v-if="reasonError" class="reason-error">{{ reasonError }}</p>
        </div>
        <p class="abstract">{{ displayAbstract }}</p>
        <button
            v-if="abstractNeedCollapse"
            type="button"
            class="toggle-btn"
            @click="abstractExpanded = !abstractExpanded"
        >
            {{ abstractExpanded ? '收起摘要' : '展开摘要' }}
        </button>

        <section v-if="semanticUnits.length" class="evidence-grid">
            <div class="evidence-toolbar">
                <p class="evidence-title">证据片段</p>
                <button
                    type="button"
                    class="toggle-btn"
                    @click="evidenceExpanded = !evidenceExpanded"
                >
                    {{ evidenceExpanded ? '收起证据' : `展开证据片段 (${semanticUnits.length})` }}
                </button>
            </div>
            <EvidenceCard
                v-for="(unit, idx) in displayEvidenceUnits"
                :key="`${normalizedId}-${idx}`"
                :unit="unit"
                :index="idx"
            />
            <p v-if="!evidenceExpanded" class="evidence-hint">
                默认隐藏全部证据片段，点击后完整展开
            </p>
        </section>
    </article>
</template>

<style scoped>
.paper-card {
    background: #ffffffee;
    border: 1px solid #c6d9de;
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(8, 49, 59, 0.06);
}

.paper-head h3 {
    margin: 0 0 0.55rem;
    color: #153f47;
    font-size: 1.1rem;
    line-height: 1.35;
}

.meta-line {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    color: #2f5860;
    background: #eef6f8;
    border: 1px solid #d7e7ea;
    border-radius: 999px;
    padding: 0.14rem 0.56rem;
}

.meta-chip.venue {
    background: #fef4e9;
    border-color: #f1d7bb;
    color: #7b4f13;
}

.meta-chip.ccf {
    background: #edf4ea;
    border-color: #cfe3c5;
    color: #366124;
}

.download-row {
    margin: 0.6rem 0 0;
    display: flex;
    gap: 0.85rem;
    flex-wrap: wrap;
}

.download-link {
    color: #0f5f6f;
    font-size: 0.88rem;
    font-weight: 700;
    text-decoration: none;
    border-bottom: 1px solid rgba(15, 95, 111, 0.4);
}

.download-link:hover {
    color: #0a3f49;
    border-bottom-color: rgba(10, 63, 73, 0.8);
}

.reason {
    margin: 0.8rem 0 0.4rem;
    color: #234046;
}

.reason-action-row {
    margin-top: 0.8rem;
}

.reason-error {
    margin: 0.42rem 0 0;
    color: #a11b1b;
    font-size: 0.86rem;
}

.abstract {
    margin: 0;
    color: #2c464d;
    line-height: 1.62;
}

.toggle-btn {
    margin-top: 0.5rem;
    border: 1px solid #b9d0d6;
    background: #f7fbfc;
    color: #0f5f6f;
    padding: 0.26rem 0.66rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    cursor: pointer;
}

.toggle-btn:hover {
    background: #edf6f8;
}

.evidence-grid {
    margin-top: 0.86rem;
    display: grid;
    gap: 0.62rem;
}

.evidence-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
    flex-wrap: wrap;
}

.evidence-title {
    margin: 0;
    color: #2b4d54;
    font-size: 0.9rem;
    font-weight: 700;
}

.evidence-hint {
    margin: 0;
    color: #4d6970;
    font-size: 0.82rem;
}
</style>
