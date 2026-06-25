<script setup>
import { computed } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps({
    unit: {
        type: Object,
        default: () => ({}),
    },
    index: {
        type: Number,
        default: 0,
    },
})

const escapeHtml = (text) =>
    String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')

const renderMath = (source, displayMode) => {
    try {
        return katex.renderToString(source, {
            displayMode,
            throwOnError: false,
            strict: 'ignore',
        })
    } catch {
        return escapeHtml(displayMode ? `$$${source}$$` : `$${source}$`)
    }
}

const renderEvidenceContent = (content) => {
    const raw = String(content || '')
    const pattern = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g
    let cursor = 0
    let output = ''

    for (const match of raw.matchAll(pattern)) {
        const matchedText = match[0]
        const blockFormula = match[1]
        const inlineFormula = match[2]
        const startIndex = match.index ?? 0

        output += escapeHtml(raw.slice(cursor, startIndex)).replace(/\n/g, '<br>')
        output += renderMath(blockFormula ?? inlineFormula ?? matchedText, Boolean(blockFormula))
        cursor = startIndex + matchedText.length
    }

    output += escapeHtml(raw.slice(cursor)).replace(/\n/g, '<br>')
    return output
}

const renderedContent = computed(() => renderEvidenceContent(props.unit?.content || ''))
</script>

<template>
    <article class="evidence-card">
        <p class="evidence-head">证据片段 {{ index + 1 }}</p>
        <p class="evidence-role">{{ unit.role || 'other' }}</p>
        <div class="evidence-content" v-html="renderedContent"></div>
    </article>
</template>

<style scoped>
.evidence-card {
    border: 1px solid #d2e3e8;
    border-radius: 12px;
    padding: 0.72rem;
    background: #f8fcfd;
}

.evidence-head {
    margin: 0;
    color: #2f606a;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.evidence-role {
    margin: 0.28rem 0 0.36rem;
    display: inline-block;
    color: #0f5561;
    background: #e5f3f6;
    border-radius: 999px;
    font-size: 0.72rem;
    padding: 0.12rem 0.5rem;
    font-weight: 700;
}

.evidence-content {
    margin: 0;
    color: #29454c;
    line-height: 1.55;
    font-size: 0.9rem;
}

.evidence-content :deep(.katex-display) {
    margin: 0.8rem 0;
    overflow-x: auto;
    overflow-y: hidden;
}

.evidence-content :deep(.katex) {
    font-size: 1.02em;
}
</style>
