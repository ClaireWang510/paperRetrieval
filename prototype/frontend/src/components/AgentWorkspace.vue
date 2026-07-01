<script setup>
import axios from 'axios'
import { computed, onMounted, ref } from 'vue'

const workspaces = ref([])
const selectedWorkspaceId = ref('')
const workspaceDetail = ref(null)
const expandedWorkspaceIds = ref({})
const showCreateWorkspaceForm = ref(false)
const loading = ref(false)
const error = ref('')

const creating = ref(false)
const newWorkspaceName = ref('')
const newWorkspaceDescription = ref('')

const uploadFile = ref(null)
const uploadTitle = ref('')
const uploading = ref(false)
const uploadError = ref('')

const arxivId = ref('')
const ingesting = ref(false)
const ingestError = ref('')
const taskMap = ref({})
const pollingTaskIds = new Set()

const chatMessage = ref('')
const chatLoading = ref(false)
const chatError = ref('')

const currentView = ref('chat')
const selectedDocumentId = ref('')

const selectedWorkspace = computed(() => workspaceDetail.value)
const documents = computed(() => (Array.isArray(selectedWorkspace.value?.documents) ? selectedWorkspace.value.documents : []))
const conversation = computed(() => (Array.isArray(selectedWorkspace.value?.conversation) ? selectedWorkspace.value.conversation : []))
const selectedDocument = computed(() => documents.value.find((doc) => doc?.id === selectedDocumentId.value) || null)

const workspaceCards = computed(() => workspaces.value.map((ws) => ({
    ...ws,
    documentCount: Array.isArray(ws.documents) ? ws.documents.length : 0,
    isExpanded: !!expandedWorkspaceIds.value[ws.id],
    isActive: ws.id === selectedWorkspaceId.value,
})))

const normalizedSemantics = computed(() => {
    const values = selectedDocument.value?.semantic_result?.normalized_semantics
    return Array.isArray(values) ? values : []
})

const semanticUnits = computed(() => {
    const values = selectedDocument.value?.semantic_result?.semantic_units
    return Array.isArray(values) ? values : []
})

const workspaceStats = computed(() => ({
    count: workspaces.value.length,
    docs: documents.value.length,
    completed: documents.value.filter((doc) => doc?.semantic_status === 'completed').length,
}))

const activeTaskCount = computed(() => Object.values(taskMap.value).filter((task) => {
    const status = task?.status
    return status && status !== 'completed' && status !== 'failed'
}).length)

const getDocumentSnippet = (doc) => {
    if (!doc) return ''
    const result = doc.semantic_result || {}
    const units = Array.isArray(result.semantic_units) ? result.semantic_units : []
    const firstUnit = units[0]
    const snippet = firstUnit?.content || firstUnit?.title || result.paper_title || ''
    if (snippet) {
        return snippet.replace(/\s+/g, ' ').slice(0, 110)
    }
    if (doc.semantic_status === 'running' || doc.semantic_status === 'queued') {
        return '语义抽取进行中'
    }
    return '等待语义抽取结果'
}

const getTaskStage = (task) => task?.stage || task?.status || 'unknown'

const syncWorkspaceInList = (workspace) => {
    if (!workspace?.id) return
    const nextItems = workspaces.value.slice()
    const index = nextItems.findIndex((item) => item?.id === workspace.id)
    if (index >= 0) nextItems[index] = workspace
    else nextItems.unshift(workspace)
    workspaces.value = nextItems
}

const ensureWorkspaceSelectionState = () => {
    expandedWorkspaceIds.value = {
        ...expandedWorkspaceIds.value,
        [selectedWorkspaceId.value]: true,
    }
    if (selectedDocumentId.value && documents.value.some((doc) => doc?.id === selectedDocumentId.value)) return
    selectedDocumentId.value = documents.value[0]?.id || ''
}

const refreshWorkspaces = async () => {
    loading.value = true
    error.value = ''
    try {
        const resp = await axios.get('/api/agent/workspaces')
        const data = resp?.data || {}
        workspaces.value = Array.isArray(data.workspaces) ? data.workspaces : []
        if (!selectedWorkspaceId.value && workspaces.value.length > 0) {
            selectedWorkspaceId.value = workspaces.value[0].id
        }
        if (selectedWorkspaceId.value) {
            await refreshWorkspaceDetail(selectedWorkspaceId.value)
        }
    } catch (e) {
        error.value = e?.response?.data?.error || e?.message || '获取知识空间列表失败'
    } finally {
        loading.value = false
    }
}

const refreshWorkspaceDetail = async (workspaceId) => {
    if (!workspaceId) return
    try {
        const resp = await axios.get(`/api/agent/workspaces/${encodeURIComponent(workspaceId)}`)
        workspaceDetail.value = resp?.data?.workspace || null
        selectedWorkspaceId.value = workspaceId
        if (workspaceDetail.value) syncWorkspaceInList(workspaceDetail.value)
        ensureWorkspaceSelectionState()

        const docs = Array.isArray(workspaceDetail.value?.documents) ? workspaceDetail.value.documents : []
        docs.forEach((doc) => {
            const taskId = doc?.semantic_task_id
            const semanticStatus = doc?.semantic_status
            if (taskId && semanticStatus && semanticStatus !== 'completed' && semanticStatus !== 'failed') {
                pollTaskUntilDone(taskId)
            }
        })
    } catch (e) {
        error.value = e?.response?.data?.error || e?.message || '获取知识空间详情失败'
    }
}

const selectWorkspace = async (workspaceId, nextView = 'chat') => {
    if (!workspaceId) return
    chatError.value = ''
    uploadError.value = ''
    ingestError.value = ''
    currentView.value = nextView
    await refreshWorkspaceDetail(workspaceId)
}

const toggleWorkspace = async (workspaceId) => {
    const nextExpanded = !expandedWorkspaceIds.value[workspaceId]
    expandedWorkspaceIds.value = { ...expandedWorkspaceIds.value, [workspaceId]: nextExpanded }
    if (selectedWorkspaceId.value !== workspaceId || !workspaceDetail.value) {
        await selectWorkspace(workspaceId, currentView.value)
    }
}

const openDocument = async (workspaceId, documentId) => {
    if (!workspaceId || !documentId) return
    if (selectedWorkspaceId.value !== workspaceId) {
        await selectWorkspace(workspaceId, 'document')
    }
    selectedDocumentId.value = documentId
    currentView.value = 'document'
}

const createWorkspace = async () => {
    if (!newWorkspaceName.value.trim() || creating.value) return
    creating.value = true
    error.value = ''
    try {
        const resp = await axios.post('/api/agent/workspaces', {
            name: newWorkspaceName.value.trim(),
            description: newWorkspaceDescription.value.trim(),
        })
        const created = resp?.data?.workspace
        newWorkspaceName.value = ''
        newWorkspaceDescription.value = ''
        showCreateWorkspaceForm.value = false
        await refreshWorkspaces()
        if (created?.id) {
            expandedWorkspaceIds.value = { ...expandedWorkspaceIds.value, [created.id]: true }
            await selectWorkspace(created.id)
        }
    } catch (e) {
        error.value = e?.response?.data?.error || e?.message || '创建知识空间失败'
    } finally {
        creating.value = false
    }
}

const onPickUpload = (event) => {
    const files = event?.target?.files
    uploadFile.value = files && files[0] ? files[0] : null
}

const uploadPdf = async () => {
    if (!selectedWorkspaceId.value || !uploadFile.value || uploading.value) return
    uploading.value = true
    uploadError.value = ''
    try {
        const form = new FormData()
        form.append('file', uploadFile.value)
        if (uploadTitle.value.trim()) form.append('title', uploadTitle.value.trim())
        const resp = await axios.post(
            `/api/agent/workspaces/${encodeURIComponent(selectedWorkspaceId.value)}/upload_pdf`,
            form,
            { headers: { 'Content-Type': 'multipart/form-data' } },
        )
        const semanticTask = resp?.data?.semantic_task
        const document = resp?.data?.document
        uploadFile.value = null
        uploadTitle.value = ''
        await refreshWorkspaceDetail(selectedWorkspaceId.value)
        if (document?.id) selectedDocumentId.value = document.id
        if (semanticTask?.id) {
            taskMap.value = { ...taskMap.value, [semanticTask.id]: semanticTask }
            pollTaskUntilDone(semanticTask.id)
        }
    } catch (e) {
        uploadError.value = e?.response?.data?.error || e?.message || '上传 PDF 失败'
    } finally {
        uploading.value = false
    }
}

const pollTaskUntilDone = async (taskId) => {
    if (!taskId || pollingTaskIds.has(taskId)) return
    pollingTaskIds.add(taskId)
    let keepPolling = true
    while (keepPolling) {
        try {
            const resp = await axios.get(`/api/agent/tasks/${encodeURIComponent(taskId)}`)
            const task = resp?.data?.task || null
            if (task) {
                taskMap.value = { ...taskMap.value, [taskId]: task }
            }
            const status = task?.status || ''
            const semanticTaskId = task?.result?.semantic_task_id
            if (semanticTaskId) pollTaskUntilDone(semanticTaskId)
            if (status === 'completed' || status === 'failed') {
                keepPolling = false
                await refreshWorkspaceDetail(selectedWorkspaceId.value)
                break
            }
        } catch {
            keepPolling = false
            break
        }
        await new Promise((resolve) => setTimeout(resolve, 2000))
    }
    pollingTaskIds.delete(taskId)
}

const ingestArxiv = async () => {
    if (!selectedWorkspaceId.value || !arxivId.value.trim() || ingesting.value) return
    ingesting.value = true
    ingestError.value = ''
    try {
        const resp = await axios.post(
            `/api/agent/workspaces/${encodeURIComponent(selectedWorkspaceId.value)}/ingest_arxiv`,
            { arxiv_id: arxivId.value.trim() },
        )
        const task = resp?.data?.task
        arxivId.value = ''
        if (task?.id) {
            taskMap.value = { ...taskMap.value, [task.id]: task }
            pollTaskUntilDone(task.id)
        }
    } catch (e) {
        ingestError.value = e?.response?.data?.error || e?.message || 'arXiv 下载任务提交失败'
    } finally {
        ingesting.value = false
    }
}

const sendChat = async () => {
    if (!selectedWorkspaceId.value || !chatMessage.value.trim() || chatLoading.value) return
    const message = chatMessage.value.trim()
    chatLoading.value = true
    chatError.value = ''
    chatMessage.value = ''
    try {
        const resp = await axios.post(
            `/api/agent/workspaces/${encodeURIComponent(selectedWorkspaceId.value)}/chat`,
            { message },
        )
        workspaceDetail.value = resp?.data?.workspace || workspaceDetail.value
        if (workspaceDetail.value) syncWorkspaceInList(workspaceDetail.value)
        currentView.value = 'chat'
    } catch (e) {
        chatError.value = e?.response?.data?.error || e?.message || 'Agent 对话失败'
        chatMessage.value = message
    } finally {
        chatLoading.value = false
    }
}

onMounted(async () => {
    await refreshWorkspaces()
})
</script>

<template>
    <section class="agent-space">
        <header class="workspace-hero">
            <div>
                <p class="eyebrow">Personal Research Space</p>
                <h2>个人学术空间</h2>
            </div>
            <div class="hero-stats">
                <div class="stat-card">
                    <span class="stat-label">知识空间</span>
                    <strong>{{ workspaceStats.count }}</strong>
                </div>
                <div class="stat-card">
                    <span class="stat-label">当前论文</span>
                    <strong>{{ workspaceStats.docs }}</strong>
                </div>
                <div class="stat-card">
                    <span class="stat-label">已完成语义抽取</span>
                    <strong>{{ workspaceStats.completed }}</strong>
                </div>
            </div>
        </header>

        <p v-if="error" class="error global-error">{{ error }}</p>

        <div class="workspace-shell">
            <aside class="workspace-sidebar">
                <section class="sidebar-card spaces-card">
                    <div class="section-head">
                        <div>
                            <p class="section-tag">Spaces</p>
                            <h3>知识空间</h3>
                        </div>
                        <button class="icon-btn" @click="showCreateWorkspaceForm = !showCreateWorkspaceForm">＋</button>
                    </div>

                    <div v-if="showCreateWorkspaceForm" class="create-inline">
                        <label class="field-label">空间名称</label>
                        <input v-model="newWorkspaceName" placeholder="例如：多智能体代码评测" />
                        <label class="field-label">空间说明</label>
                        <textarea
                            v-model="newWorkspaceDescription"
                            rows="3"
                            placeholder="例如：聚焦 agent 系统评估、工具调用稳定性与多轮协同"
                        />
                        <div class="row-actions">
                            <button class="action-btn primary" :disabled="creating" @click="createWorkspace">
                                {{ creating ? '创建中...' : '新建知识空间' }}
                            </button>
                            <button class="action-btn subtle" @click="showCreateWorkspaceForm = false">取消</button>
                        </div>
                    </div>

                    <div v-if="!loading && workspaceCards.length === 0" class="empty-state">
                        还没有知识空间，先点加号新建一个。
                    </div>

                    <div class="workspace-list">
                        <article
                            v-for="ws in workspaceCards"
                            :key="ws.id"
                            class="workspace-item"
                            :class="{ active: ws.isActive }"
                        >
                            <button class="workspace-head" @click="toggleWorkspace(ws.id)">
                                <div class="workspace-head-text">
                                    <p class="workspace-title">{{ ws.name || '未命名空间' }}</p>
                                    <p class="workspace-desc">{{ ws.description || '暂无介绍' }}</p>
                                </div>
                                <span class="workspace-toggle">{{ ws.isExpanded ? '−' : '+' }}</span>
                            </button>

                            <div v-if="ws.isExpanded" class="workspace-body">
                                <div class="workspace-meta-row">
                                    <span>{{ ws.documentCount }} 篇论文</span>
                                    <button class="mini-link" @click="selectWorkspace(ws.id, 'chat')">打开空间</button>
                                </div>
                            </div>
                        </article>
                    </div>
                </section>
            </aside>

            <section class="workspace-stage">
                <template v-if="selectedWorkspace">
                    <header class="stage-header">
                        <div>
                            <p class="section-tag">Workspace</p>
                            <h3>{{ selectedWorkspace.name }}</h3>
                            <p class="stage-copy">{{ selectedWorkspace.description || '暂无介绍' }}</p>
                        </div>
                        <div class="stage-actions">
                            <button class="view-pill" :class="{ active: currentView === 'chat' }" @click="currentView = 'chat'">对话</button>
                            <button class="view-pill" :class="{ active: currentView === 'document' }" :disabled="!selectedDocument" @click="currentView = 'document'">论文页</button>
                            <button class="action-btn subtle" @click="refreshWorkspaceDetail(selectedWorkspaceId)">刷新</button>
                        </div>
                    </header>

                    <div class="workspace-body-grid">
                        <section class="main-stage">
                            <template v-if="currentView === 'chat'">
                                <div class="chat-frame">
                                    <div class="messages">
                                        <p v-if="conversation.length === 0" class="muted intro-message">在这里向 Agent 提问。</p>
                                        <article
                                            v-for="turn in conversation"
                                            :key="turn.id"
                                            class="msg"
                                            :class="turn.role === 'user' ? 'user' : 'assistant'"
                                        >
                                            <p class="role">{{ turn.role === 'user' ? '你' : 'Agent' }}</p>
                                            <p class="content">{{ turn.content }}</p>
                                        </article>
                                    </div>

                                    <div class="composer">
                                        <textarea
                                            v-model="chatMessage"
                                            rows="4"
                                            placeholder="输入你的问题..."
                                        />
                                        <div class="composer-footer">
                                            <p v-if="chatError" class="error">{{ chatError }}</p>
                                            <button class="action-btn primary" :disabled="chatLoading || !chatMessage.trim()" @click="sendChat">
                                                {{ chatLoading ? '回复中...' : '发送' }}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </template>

                            <template v-else>
                                <article v-if="selectedDocument" class="document-page">
                                    <header class="document-head">
                                        <div>
                                            <p class="section-tag">Paper View</p>
                                            <h4>{{ selectedDocument.title || selectedDocument.filename || selectedDocument.arxiv_id || '未命名论文' }}</h4>
                                            <p class="document-meta">{{ selectedDocument.kind }} · {{ selectedDocument.semantic_status || 'pending' }}</p>
                                        </div>
                                        <button class="action-btn subtle" @click="currentView = 'chat'">返回对话</button>
                                    </header>

                                    <p v-if="selectedDocument.semantic_error" class="error">{{ selectedDocument.semantic_error }}</p>

                                    <section v-if="selectedDocument.semantic_result" class="summary-grid">
                                        <article class="summary-card accent">
                                            <span class="summary-label">论文标题</span>
                                            <strong>{{ selectedDocument.semantic_result.paper_title || selectedDocument.semantic_result.paper_id }}</strong>
                                        </article>
                                        <article class="summary-card">
                                            <span class="summary-label">原子事实</span>
                                            <strong>{{ selectedDocument.semantic_result.atomic_fact_count || 0 }}</strong>
                                        </article>
                                        <article class="summary-card">
                                            <span class="summary-label">语义单元</span>
                                            <strong>{{ selectedDocument.semantic_result.semantic_unit_count || 0 }}</strong>
                                        </article>
                                    </section>

                                    <section v-if="normalizedSemantics.length" class="semantic-overview">
                                        <div class="section-head">
                                            <div>
                                                <p class="section-tag">Buckets</p>
                                                <h5>语义标签</h5>
                                            </div>
                                        </div>
                                        <div class="semantic-pills">
                                            <span
                                                v-for="bucket in normalizedSemantics.slice(0, 10)"
                                                :key="`${selectedDocument.id}-${bucket.label}`"
                                                class="semantic-pill"
                                            >
                                                {{ bucket.label }} ({{ bucket.fact_count }})
                                            </span>
                                        </div>
                                    </section>

                                    <section v-if="semanticUnits.length" class="semantic-units-section">
                                        <div class="section-head">
                                            <div>
                                                <p class="section-tag">Units</p>
                                                <h5>语义表示</h5>
                                            </div>
                                        </div>
                                        <div class="unit-grid">
                                            <article
                                                v-for="unit in semanticUnits"
                                                :key="unit.id || `${selectedDocument.id}-${unit.cluster_index}`"
                                                class="unit-card"
                                            >
                                                <p class="unit-role">{{ unit.normalized_semantic_role || unit.semantic_role || 'other' }}</p>
                                                <h6>{{ unit.title || 'Untitled unit' }}</h6>
                                                <p class="unit-content">{{ unit.content }}</p>
                                                <div class="keyword-row" v-if="(unit.keywords || []).length">
                                                    <span v-for="keyword in unit.keywords.slice(0, 8)" :key="`${unit.id}-${keyword}`" class="keyword-pill">{{ keyword }}</span>
                                                </div>
                                            </article>
                                        </div>
                                    </section>

                                    <section v-if="normalizedSemantics.length" class="fact-snapshot-section">
                                        <div class="section-head">
                                            <div>
                                                <p class="section-tag">Facts</p>
                                                <h5>样例事实</h5>
                                            </div>
                                        </div>
                                        <div class="bucket-grid">
                                            <article
                                                v-for="bucket in normalizedSemantics.slice(0, 4)"
                                                :key="`${selectedDocument.id}-bucket-${bucket.label}`"
                                                class="bucket-card"
                                            >
                                                <p class="bucket-name">{{ bucket.label }}</p>
                                                <p
                                                    v-for="fact in (bucket.sample_facts || []).slice(0, 3)"
                                                    :key="`${selectedDocument.id}-${bucket.label}-${fact}`"
                                                    class="fact-item"
                                                >
                                                    {{ fact }}
                                                </p>
                                            </article>
                                        </div>
                                    </section>
                                </article>

                                <div v-else class="empty-document-page">
                                    <p class="muted">先从右侧论文列表中选择一篇论文。</p>
                                </div>
                            </template>
                        </section>

                        <aside class="paper-rail">
                            <section class="rail-card">
                                <div class="section-head">
                                    <div>
                                        <p class="section-tag">Papers</p>
                                        <h4>论文列表</h4>
                                    </div>
                                </div>

                                <p v-if="documents.length === 0" class="muted small">当前空间还没有论文。</p>

                                <div class="paper-list">
                                    <button
                                        v-for="doc in documents"
                                        :key="doc.id"
                                        class="paper-item"
                                        :class="{ active: doc.id === selectedDocumentId }"
                                        @click="openDocument(selectedWorkspaceId, doc.id)"
                                    >
                                        <span class="paper-title">{{ doc.title || doc.filename || doc.arxiv_id || '未命名论文' }}</span>
                                        <span class="paper-snippet">{{ getDocumentSnippet(doc) }}</span>
                                        <span class="paper-meta">{{ doc.semantic_status || 'pending' }}</span>
                                    </button>
                                </div>
                            </section>

                            <section class="rail-card">
                                <div class="section-head">
                                    <div>
                                        <p class="section-tag">Ingest</p>
                                        <h4>导入</h4>
                                    </div>
                                </div>

                                <div class="stack-fields">
                                    <label class="field-label">上传 PDF</label>
                                    <input type="file" accept="application/pdf" @change="onPickUpload" />
                                    <input v-model="uploadTitle" placeholder="可选：文档标题" />
                                    <button class="action-btn primary" :disabled="uploading || !uploadFile" @click="uploadPdf">
                                        {{ uploading ? '上传中...' : '上传 PDF' }}
                                    </button>
                                    <p v-if="uploadError" class="error">{{ uploadError }}</p>
                                </div>

                                <div class="divider" />

                                <div class="stack-fields">
                                    <label class="field-label">导入 arXiv</label>
                                    <input v-model="arxivId" placeholder="例如：2405.12345" />
                                    <button class="action-btn" :disabled="ingesting || !arxivId.trim()" @click="ingestArxiv">
                                        {{ ingesting ? '提交中...' : '下载 arXiv' }}
                                    </button>
                                    <p v-if="ingestError" class="error">{{ ingestError }}</p>
                                </div>
                            </section>

                            <section class="rail-card">
                                <div class="section-head">
                                    <div>
                                        <p class="section-tag">Tasks</p>
                                        <h4>处理任务</h4>
                                    </div>
                                </div>
                                <p class="muted small">{{ activeTaskCount }} 个任务进行中</p>
                                <ul class="task-list" v-if="Object.keys(taskMap).length">
                                    <li v-for="(task, taskId) in taskMap" :key="taskId" class="task-row">
                                        <span class="task-name">{{ task.type }}</span>
                                        <span class="task-stage">{{ getTaskStage(task) }}</span>
                                    </li>
                                </ul>
                            </section>
                        </aside>
                    </div>
                </template>

                <div v-else class="empty-stage">
                    <p class="muted">请先从左侧新建或选择一个知识空间。</p>
                </div>
            </section>
        </div>
    </section>
</template>

<style scoped>
.agent-space {
    --surface: rgba(255, 252, 246, 0.92);
    --surface-strong: rgba(255, 255, 255, 0.96);
    --line: rgba(18, 80, 92, 0.14);
    --line-strong: rgba(18, 80, 92, 0.28);
    --text: #17353c;
    --muted: #567278;
    --accent: #165a66;
    --accent-soft: #dff0f1;
    --accent-strong: #0f4f59;
    --warm: #c98a2b;
    display: grid;
    gap: 1rem;
}

.workspace-hero {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-end;
    padding: 1.1rem 1.2rem;
    border-radius: 24px;
    border: 1px solid rgba(201, 138, 43, 0.16);
    background:
        radial-gradient(circle at 8% 20%, rgba(201, 138, 43, 0.14), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(22, 90, 102, 0.18), transparent 30%),
        linear-gradient(145deg, #fffaf2 0%, #f4fbfc 58%, #ffffff 100%);
    box-shadow: 0 18px 48px rgba(28, 55, 61, 0.08);
}

.eyebrow,
.section-tag,
.summary-label,
.stat-label,
.unit-role {
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.74rem;
    font-weight: 700;
}

.eyebrow,
.section-tag,
.summary-label {
    color: var(--accent);
}

.workspace-hero h2,
.stage-header h3,
.document-head h4,
.chat-frame h4,
.rail-card h4 {
    margin: 0.3rem 0 0;
    color: #102f35;
    font-family: 'Source Han Serif SC', 'Noto Serif SC', serif;
}

.hero-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    min-width: 320px;
}

.stat-card,
.summary-card {
    display: grid;
    gap: 0.25rem;
    padding: 0.95rem;
    border-radius: 18px;
    border: 1px solid rgba(18, 80, 92, 0.12);
    background: rgba(255, 255, 255, 0.72);
}

.stat-card strong,
.summary-card strong {
    color: #153b42;
    font-size: clamp(1.15rem, 2vw, 1.55rem);
}

.stat-label {
    color: #6b7f84;
}

.workspace-shell {
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr);
    gap: 1rem;
    align-items: start;
}

.workspace-sidebar {
    display: grid;
    gap: 0.9rem;
    position: sticky;
    top: 1rem;
}

.sidebar-card,
.workspace-stage,
.rail-card,
.document-page,
.empty-stage,
.empty-document-page {
    border-radius: 22px;
    border: 1px solid var(--line);
    background: var(--surface);
    box-shadow: 0 16px 42px rgba(27, 51, 58, 0.08);
}

.sidebar-card,
.rail-card,
.document-page,
.empty-stage,
.empty-document-page {
    padding: 1rem;
}

.workspace-stage {
    min-width: 0;
    padding: 1rem;
}

.section-head,
.stage-header,
.document-head {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: flex-start;
}

.stage-header,
.document-head {
    margin-bottom: 0.85rem;
}

.field-label {
    display: inline-block;
    margin: 0.4rem 0 0.25rem;
    color: #3f6067;
    font-size: 0.84rem;
    font-weight: 600;
}

input,
textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid rgba(21, 80, 90, 0.18);
    border-radius: 14px;
    padding: 0.7rem 0.78rem;
    font-size: 0.95rem;
    color: var(--text);
    background: rgba(255, 255, 255, 0.92);
    transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

input:focus,
textarea:focus {
    outline: none;
    border-color: rgba(22, 90, 102, 0.55);
    box-shadow: 0 0 0 4px rgba(22, 90, 102, 0.12);
    background: #ffffff;
}

.action-btn,
.view-pill,
.workspace-head,
.paper-item,
.icon-btn,
.mini-link {
    transition: transform 140ms ease, background 140ms ease, border-color 140ms ease, color 140ms ease, opacity 140ms ease;
}

.action-btn {
    border: 1px solid rgba(21, 80, 90, 0.16);
    border-radius: 999px;
    background: #f1f7f7;
    color: var(--accent-strong);
    padding: 0.62rem 0.98rem;
    cursor: pointer;
    font-weight: 700;
}

.action-btn.primary {
    border: none;
    color: #fffef9;
    background: linear-gradient(118deg, #155a66, #2f7c89);
}

.action-btn.subtle {
    background: #fff8eb;
    border-color: rgba(201, 138, 43, 0.18);
    color: #83551a;
}

.icon-btn {
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 999px;
    border: 1px solid rgba(21, 80, 90, 0.16);
    background: linear-gradient(118deg, #155a66, #2f7c89);
    color: #fffef9;
    font-size: 1.15rem;
    font-weight: 700;
    cursor: pointer;
}

.action-btn:hover,
.view-pill:hover,
.workspace-head:hover,
.paper-item:hover,
.icon-btn:hover,
.mini-link:hover {
    transform: translateY(-1px);
}

.action-btn:disabled,
.view-pill:disabled,
.paper-item:disabled {
    opacity: 0.62;
    cursor: wait;
    transform: none;
}

.workspace-list {
    display: grid;
    gap: 0.65rem;
}

.workspace-item {
    border: 1px solid rgba(21, 80, 90, 0.12);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.76);
    overflow: hidden;
}

.workspace-item.active {
    border-color: rgba(22, 90, 102, 0.32);
}

.workspace-head {
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: center;
    text-align: left;
    border: none;
    background: transparent;
    padding: 0.82rem 0.9rem;
    cursor: pointer;
}

.workspace-head-text {
    min-width: 0;
}

.workspace-title {
    margin: 0;
    color: #163c43;
    font-weight: 700;
}

.workspace-desc {
    margin: 0.18rem 0 0;
    color: var(--muted);
    font-size: 0.83rem;
    line-height: 1.4;
}

.workspace-toggle {
    color: var(--accent);
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1;
}

.workspace-body {
    padding: 0 0.9rem 0.9rem;
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: center;
}

.workspace-meta-row {
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 0.7rem;
    align-items: center;
    color: var(--muted);
    font-size: 0.84rem;
}

.mini-link {
    width: fit-content;
    padding: 0;
    border: none;
    background: transparent;
    color: #8a5d20;
    cursor: pointer;
    font-size: 0.84rem;
    font-weight: 700;
}

.workspace-body-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 1rem;
}

.main-stage {
    min-width: 0;
}

.paper-rail {
    display: grid;
    gap: 0.9rem;
    align-content: start;
}

.create-inline,
.paper-list,
.task-list,
.semantic-pills,
.keyword-row,
.bucket-grid,
.unit-grid,
.summary-grid {
    display: grid;
    gap: 0.6rem;
}

.paper-list {
    max-height: 28rem;
    overflow-y: auto;
    padding-right: 0.15rem;
}

.paper-item {
    width: 100%;
    text-align: left;
    border: 1px solid rgba(21, 80, 90, 0.12);
    border-radius: 16px;
    background: #f8fbfb;
    padding: 0.78rem 0.82rem;
    cursor: pointer;
    display: grid;
    gap: 0.22rem;
}

.paper-item.active {
    border-color: rgba(22, 90, 102, 0.4);
    background: #eaf5f5;
}

.paper-title {
    color: #14373e;
    font-weight: 700;
    line-height: 1.35;
}

.paper-snippet,
.paper-meta,
.stage-copy,
.document-meta,
.muted,
.small {
    color: var(--muted);
}

.paper-snippet,
.paper-meta,
.document-meta,
.small {
    font-size: 0.82rem;
    line-height: 1.45;
}

.chat-frame {
    display: grid;
    gap: 0.9rem;
    min-height: 70vh;
    border-radius: 22px;
    border: 1px solid var(--line);
    background: var(--surface-strong);
    padding: 1rem;
}

.messages {
    flex: 1;
    min-height: 48vh;
    max-height: 60vh;
    overflow-y: auto;
    border: 1px solid rgba(21, 80, 90, 0.1);
    border-radius: 18px;
    padding: 1rem;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(247, 251, 251, 0.96));
    display: grid;
    gap: 0.75rem;
}

.intro-message {
    align-self: center;
    justify-self: center;
    max-width: 28rem;
    text-align: center;
}

.msg {
    max-width: min(78%, 48rem);
    border-radius: 18px;
    padding: 0.82rem 0.95rem;
}

.msg.user {
    justify-self: end;
    background: linear-gradient(135deg, #e4f2f4, #eef8f8);
    border: 1px solid rgba(22, 90, 102, 0.16);
}

.msg.assistant {
    justify-self: start;
    background: linear-gradient(135deg, #fffaf1, #fffdf8);
    border: 1px solid rgba(201, 138, 43, 0.18);
}

.role {
    margin: 0 0 0.26rem;
    color: #506b70;
    font-size: 0.79rem;
    font-weight: 700;
}

.content,
.unit-content,
.fact-item {
    margin: 0;
    color: #1a3940;
    white-space: pre-wrap;
    line-height: 1.65;
}

.composer {
    display: grid;
    gap: 0.65rem;
}

.composer textarea {
    min-height: 7rem;
    resize: vertical;
    border-radius: 18px;
}

.composer-footer,
.stage-actions {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: center;
    flex-wrap: wrap;
}

.view-pill {
    border: 1px solid rgba(21, 80, 90, 0.16);
    border-radius: 999px;
    background: #f6fbfb;
    color: #1b525d;
    padding: 0.55rem 0.88rem;
    font-weight: 700;
    cursor: pointer;
}

.view-pill.active {
    border-color: rgba(22, 90, 102, 0.34);
    background: linear-gradient(118deg, #155a66, #2f7c89);
    color: #fffef9;
}

.summary-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.summary-card.accent {
    border-color: rgba(201, 138, 43, 0.22);
    background: linear-gradient(145deg, #fff9ef, #ffffff);
}

.semantic-overview,
.semantic-units-section,
.fact-snapshot-section,
.document-page {
    border: 1px solid rgba(21, 80, 90, 0.12);
    border-radius: 22px;
    padding: 1rem;
    background: #ffffff;
}

.semantic-pill,
.keyword-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.22rem 0.62rem;
    font-size: 0.78rem;
    font-weight: 700;
}

.semantic-pill {
    border: 1px solid rgba(21, 80, 90, 0.14);
    background: #edf6f7;
    color: #1c5661;
}

.keyword-pill {
    border: 1px solid rgba(201, 138, 43, 0.18);
    background: #fff8ee;
    color: #8a5d20;
}

.unit-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.unit-card,
.bucket-card {
    border: 1px solid rgba(21, 80, 90, 0.1);
    border-radius: 18px;
    padding: 0.95rem;
    background: #ffffff;
}

.unit-card h6,
.bucket-name {
    margin: 0.25rem 0 0.45rem;
    color: #163a41;
}

.bucket-name {
    font-size: 0.94rem;
    font-weight: 800;
}

.bucket-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.task-list {
    margin: 0.45rem 0 0;
}

.task-row {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
    border-radius: 12px;
    border: 1px solid rgba(21, 80, 90, 0.08);
    padding: 0.55rem 0.7rem;
    background: #f9fcfc;
}

.task-name {
    color: #163941;
    font-size: 0.86rem;
}

.task-stage {
    color: #8a5d20;
    font-size: 0.78rem;
    font-weight: 700;
}

.error {
    color: #a31f1f;
    margin: 0;
}

.global-error {
    padding: 0.85rem 1rem;
    border-radius: 16px;
    background: rgba(255, 241, 241, 0.92);
    border: 1px solid rgba(190, 48, 48, 0.16);
}

.empty-state,
.empty-stage,
.empty-document-page {
    color: var(--muted);
    line-height: 1.6;
}

.divider {
    height: 1px;
    margin: 0.95rem 0;
    background: linear-gradient(90deg, rgba(21, 80, 90, 0.02), rgba(21, 80, 90, 0.18), rgba(21, 80, 90, 0.02));
}

@media (max-width: 1180px) {
    .workspace-shell {
        grid-template-columns: 280px minmax(0, 1fr);
    }

    .workspace-body-grid {
        grid-template-columns: minmax(0, 1fr) 250px;
    }

    .unit-grid,
    .bucket-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 960px) {
    .workspace-hero,
    .workspace-shell,
    .workspace-body-grid,
    .summary-grid {
        grid-template-columns: 1fr;
    }

    .workspace-hero {
        flex-direction: column;
        align-items: stretch;
    }

    .workspace-sidebar {
        position: static;
    }

    .hero-stats {
        min-width: 0;
    }

    .stage-header,
    .document-head,
    .composer-footer,
    .stage-actions {
        flex-direction: column;
        align-items: stretch;
    }

    .msg {
        max-width: 100%;
    }
}

@media (max-width: 680px) {
    .hero-stats {
        grid-template-columns: 1fr;
    }

    .sidebar-card,
    .rail-card,
    .chat-frame,
    .document-page,
    .semantic-overview,
    .semantic-units-section,
    .fact-snapshot-section,
    .empty-stage,
    .empty-document-page {
        padding: 0.85rem;
        border-radius: 18px;
    }

    .messages {
        min-height: 38vh;
        max-height: none;
    }
}
</style>