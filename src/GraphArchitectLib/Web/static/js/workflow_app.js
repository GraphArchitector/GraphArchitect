// === Workflow Visualizer Application ===
// WebSocket connection and UI management

class WorkflowVisualizer {
    constructor() {
        this.socket = null;
        this.currentWorkflow = null;
        this.currentStepIndex = 0;
        this.startTime = null;
        this.timerInterval = null;
        this.agentsLibrary = {};
        
        this.currentExecutionBlock = null;
        this.currentExecutionContent = null;
        this.isWorkflowFinished = false;
        
        this.init();
    }
    
    init() {
        console.log('[INIT] Initializing Workflow Visualizer');
        this.connectWebSocket();
        this.loadAgentLibrary();
        this.setupEventListeners();
    }
    
    connectWebSocket() {
        this.socket = io({
            path: '/socket.io',
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10
        });
        
        this.socket.on('connect', () => {
            console.log('[OK] WebSocket connected');
            document.getElementById('stat-status').textContent = 'Connected';
        });
        
        this.socket.on('workflow_info', (data) => this.onWorkflowInfo(data));
        this.socket.on('generation_phase_started', (data) => this.onGenerationPhaseStarted(data));
        this.socket.on('generation_progress', (data) => this.onGenerationProgress(data));
        this.socket.on('generation_phase_completed', (data) => this.onGenerationPhaseCompleted(data));
        this.socket.on('step_started', (data) => this.onStepStarted(data));
        this.socket.on('agent_progress', (data) => this.onAgentProgress(data));
        this.socket.on('agent_score_updated', (data) => this.onAgentScoreUpdated(data));
        this.socket.on('agent_selected', (data) => this.onAgentSelected(data));
        this.socket.on('agent_executing', (data) => this.onAgentExecuting(data));
        this.socket.on('step_completed', (data) => this.onStepCompleted(data));
        this.socket.on('workflow_completed', (data) => this.onWorkflowCompleted(data));
        this.socket.on('workflow_stopped', (data) => this.onWorkflowStopped(data));
        this.socket.on('workflow_error', (data) => this.onWorkflowError(data));
    }
    
    async loadAgentLibrary() {
        try {
            const response = await fetch('/api/agents-library');
            const data = await response.json();
            data.agents.forEach(agent => this.agentsLibrary[agent.id] = agent);
            this.renderAgentLibrary(data.agents);
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }
    
    setupEventListeners() {
        document.getElementById('agent-search').addEventListener('input', (e) => this.filterAgentLibrary(e.target.value));
    }
    
    // === Workflow Control ===
    
    async startWorkflow(customTemplate = null, files = []) {
        const algorithm = customTemplate || document.getElementById('template-selector').value;
        if (!algorithm) return;
        
        document.getElementById('welcome-message').style.display = 'none';
        this.currentExecutionBlock = null;
        this.currentExecutionContent = null;
        this.startTime = Date.now();
        this.startTimer();
        this.isWorkflowFinished = false;
        this.setStopState(true);
        document.getElementById('stat-status').textContent = 'Running';

        try {
            const formData = new FormData();
            // Cохраненное сообщение или из input
            const userMessage = this._pendingMessage || document.getElementById('message-input').value || "Execute workflow";
            this._pendingMessage = null;  // Очистка после использования
            formData.append('message', userMessage);
            formData.append('files', JSON.stringify(files));
            formData.append('planning_algorithm', algorithm);
            formData.append('use_streaming', 'true');
            
            // ReWOO планирование
            const useRewooCheckbox = document.getElementById('use-rewoo-checkbox');
            if (useRewooCheckbox && useRewooCheckbox.checked) {
                formData.append('use_rewoo', 'true');
            }
            
            // User Priority (адаптивность)
            const prioritySelector = document.getElementById('user-priority-selector');
            if (prioritySelector) {
                formData.append('user_priority', prioritySelector.value);
            }

            const response = await fetch(`/api/chat/wf_${Date.now()}/message/stream`, {
                method: 'POST',
                body: formData
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                // Последняя строка может быть неполной — оставляем в буфере
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        this.handleApiEvent(data);
                    } catch (e) {
                        console.warn("Incomplete JSON line, buffering...", line.substring(0, 100));
                    }
                }
            }
            
            // Обрабатываем остаток буфера
            if (buffer.trim()) {
                try {
                    const data = JSON.parse(buffer);
                    this.handleApiEvent(data);
                } catch (e) {
                    console.warn("Could not parse final buffer:", buffer.substring(0, 100));
                }
            }
        } catch (error) {
            console.error('API Error:', error);
            this.onWorkflowError({ error: error.message });
        }
    }

    handleApiEvent(data) {
        console.log('[API EVENT]', data.type, data);
        
        // Маппинг типов событий API на обработчики
        switch (data.type) {
            case 'workflow_info': this.onWorkflowInfo(data.metadata); break;
            case 'gen_phase_start': this.onGenerationPhaseStarted(data); break;
            case 'gen_progress': this.onGenerationProgress(data); break;
            case 'gen_phase_complete': this.onGenerationPhaseCompleted(data); break;
            case 'step_started': {
                let stepIdx = 0;
                if (this.currentWorkflow && this.currentWorkflow.steps) {
                    const foundIdx = this.currentWorkflow.steps.findIndex(s => s.id === data.step_id);
                    stepIdx = foundIdx !== -1 ? foundIdx : (this.currentStepIndex || 0);
                }
                this.onStepStarted({
                    stepId: data.step_id,
                    stepIndex: stepIdx,
                    stepName: data.metadata ? data.metadata.name : data.step_id,
                    candidateAgents: data.metadata ? data.metadata.candidates : []
                });
                break;
            }
            case 'agent_progress': 
                this.onAgentProgress({ agentId: data.agent_id, progress: data.progress });
                break;
            case 'agent_score_updated': 
                if (data.metadata && data.metadata.agents) {
                    this.onAgentScoreUpdated({ agents: data.metadata.agents });
                }
                break;
            case 'agent_selected': 
                this.onAgentSelected({ winnerId: data.agent_id, score: data.score });
                break;
            case 'agent_executing': 
                this.onAgentExecuting({ agentId: data.agent_id, progress: data.progress, action: data.content });
                break;
            case 'step_completed': this.onStepCompleted(data); break;
            case 'text': this.onWorkflowCompleted({ finalAnswer: data.content }); break;
            case 'error': this.onWorkflowError({ error: data.content }); break;
        }
    }

    setStopState(isRunning) {
        const sendBtn = document.getElementById('send-btn');
        if (isRunning) {
            sendBtn.innerHTML = '⏹';
            sendBtn.style.background = '#ff6b6b';
            sendBtn.title = 'Остановить выполнение';
            sendBtn.type = 'button'; // Предотвращаем отправку формы
        } else {
            sendBtn.innerHTML = 'Send';
            sendBtn.style.background = 'var(--f-blue)';
            sendBtn.title = 'Отправить сообщение';
            sendBtn.type = 'submit';
        }
    }

    startWorkflowFromChat(message, files = []) {
        this.addUserMessage(message, files);
        
        // Сохраняем сообщение для передачи в API
        this._pendingMessage = message;
        
        // Используем выбранный алгоритм планирования из селектора
        const algorithm = document.getElementById('template-selector').value;
        
        this.startWorkflow(algorithm, files);
    }

    addUserMessage(text, files = []) {
        const chatMessages = document.getElementById('chat-messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        
        let content = text;
        if (files.length > 0) {
            content += `<div class="msg-files">📎 Прикреплено файлов: ${files.length}</div>`;
        }
        
        msgDiv.innerHTML = content;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    addAssistantMessage(text, agentName = "Архитектор") {
        const chatMessages = document.getElementById('chat-messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant-message';
        
        console.log('[DEBUG] Adding assistant message, length:', text.length);
        console.log('[DEBUG] Has image:', text.includes('![image]'));
        console.log('[DEBUG] Has base64:', text.includes('data:image'));
        
        // Проверка лимита браузера на data URL (Chrome ~2MB, Firefox ~32MB)
        if (text.includes('data:image') && text.length > 2000000) {
            console.warn('[WARN] Image data URL too large for Chrome:', text.length, 'bytes');
            msgDiv.innerHTML = `<strong>${agentName}:</strong><br><div class="markdown-content">
                <p style="color: #ef4444;"> Изображение слишком большое для отображения в браузере (${(text.length/1024/1024).toFixed(2)} MB)</p>
                <p>Лимит Chrome: ~2MB для data URL. Попробуйте запросить изображение меньшего размера.</p>
            </div>`;
            chatMessages.appendChild(msgDiv);
            return;
        }
        
        // Поддержка Markdown через marked.js
        let htmlContent = text;
        if (typeof marked !== 'undefined') {
            try {
                // Настройка marked для поддержки переносов строк
                marked.setOptions({
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false
                });
                
                // Для очень больших base64-изображений может потребоваться время
                if (text.length > 100000) {
                    console.log('[DEBUG] Large content detected (', (text.length/1024).toFixed(0), 'KB), parsing...');
                }
                
                htmlContent = marked.parse(text);
                console.log('[DEBUG] Markdown parsed successfully, HTML length:', htmlContent.length);
                
                // Проверяем, что изображение действительно создалось
                if (text.includes('data:image') && htmlContent.includes('<img')) {
                    console.log('[DEBUG] Image tag created successfully');
                } else if (text.includes('data:image')) {
                    console.warn('[DEBUG] Image in source but no <img> tag in output!');
                    console.warn('[DEBUG] First 200 chars of text:', text.substring(0, 200));
                    console.warn('[DEBUG] First 200 chars of HTML:', htmlContent.substring(0, 200));
                }
                
            } catch (e) {
                console.error("Marked parsing error:", e);
                // Fallback to simple newline replacement if marked fails
                htmlContent = text.replace(/\n/g, '<br>');
            }
        } else {
            console.warn('[DEBUG] Marked.js not loaded, using fallback');
            // Fallback if marked is not loaded
            htmlContent = text.replace(/\n/g, '<br>');
        }
        
        msgDiv.innerHTML = `<strong>${agentName}:</strong><br><div class="markdown-content">${htmlContent}</div>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        console.log('[DEBUG] Message added to DOM');
        
        // Дополнительная проверка: есть ли <img> в DOM?
        const imgs = msgDiv.querySelectorAll('img');
        console.log('[DEBUG] Images in DOM:', imgs.length);
        if (imgs.length > 0) {
            imgs.forEach((img, idx) => {
                console.log(`[DEBUG] Image ${idx}: src length =`, img.src.length);
                img.onload = () => console.log(`[DEBUG] Image ${idx} loaded successfully`);
                img.onerror = (e) => console.error(`[DEBUG] Image ${idx} failed to load:`, e);
            });
        }
    }

    stopWorkflow() {
        if (this.currentWorkflow) {
            this.socket.emit('stop_workflow', { workflow_id: this.currentWorkflow.workflowId });
            this.stopTimer();
            this.resetUI();
        }
    }
    
    // === Logging & Collapsible Blocks ===
    
    addLog(type, message) {
        if (!this.currentExecutionBlock || this.isWorkflowFinished) {
            this.createNewExecutionBlock();
        }

        const logEntry = document.createElement('div');
        logEntry.style.padding = '4px 0';
        logEntry.style.borderBottom = '1px solid #f1f1f1';
        logEntry.style.wordBreak = 'break-word';
        logEntry.style.overflowWrap = 'break-word';
        
        const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
        
        // Если в логе есть изображение base64, показываем превью
        let content = message;
        if (message.includes('![image](data:image')) {
            // Извлекаем только начало для превью
            const previewText = message.substring(0, 100) + '... [IMAGE]';
            content = `<span style="color: #4c6ef5; font-style: italic;">${previewText}</span>`;
        } else if (message.length > 300) {
            // Обрезаем слишком длинные логи
            content = message.substring(0, 300) + '...';
        }
        
        logEntry.innerHTML = `<span style="color: #adb5bd; margin-right: 8px;">[${time}]</span> <span>${content}</span>`;
        
        this.currentExecutionContent.appendChild(logEntry);
        this.currentExecutionContent.scrollTop = this.currentExecutionContent.scrollHeight;
        
        document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
    }

    createNewExecutionBlock() {
        const chatMessages = document.getElementById('chat-messages');
        const block = document.createElement('div');
        block.className = 'execution-block';
        
        block.innerHTML = `
            <div class="execution-header">
                <span class="exec-title" style="font-weight: 600;">Выполнение графа агентов...</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="execution-content"></div>
        `;
        
        chatMessages.appendChild(block);
        this.currentExecutionBlock = block;
        this.currentExecutionContent = block.querySelector('.execution-content');
        this.isWorkflowFinished = false;
        
        block.querySelector('.execution-header').onclick = () => {
            block.classList.toggle('collapsed');
            const icon = block.querySelector('.toggle-icon');
            icon.textContent = block.classList.contains('collapsed') ? '►' : '▼';
        };
    }

    // === WebSocket Event Handlers ===
    
    onWorkflowInfo(data) {
        this.currentWorkflow = data;
        if (data.steps && data.steps.length > 0) {
            this.renderWorkflowSteps(data.steps);
        }
        this.addLog('info', `[OK] Граф: ${data.name} (${data.steps ? data.steps.length : 0} шагов)`);
    }

    onGenerationPhaseStarted(data) {
        this.addLog('info', `[INFO] ${data.content}`);
        
        // Гарантируем наличие общего блока выполнения
        if (!this.currentExecutionBlock || this.isWorkflowFinished) {
            this.createNewExecutionBlock();
        }

        // Создаем визуальный элемент генерации ВНУТРИ лога, если его еще нет
        let genContainer = this.currentExecutionContent.querySelector('.generation-log-compact');
        if (!genContainer) {
            genContainer = document.createElement('div');
            genContainer.className = 'generation-log-compact';
            genContainer.innerHTML = `
                <div class="gen-log-phases">
                    <div class="gen-log-item" data-phase="knn">1. Поиск в k-NN <span class="gen-status">...</span></div>
                    <div class="gen-log-item" data-phase="graph_algo">2. Генерация цепочек <span class="gen-status">...</span></div>
                    <div class="gen-log-item" data-phase="llm_refine">3. Обработка LLM <span class="gen-status">...</span></div>
                </div>
                <div class="gen-log-progress-bg"><div class="gen-log-progress-fill"></div></div>
            `;
            this.currentExecutionContent.prepend(genContainer);
        }
        
        // Подсвечиваем текущую фазу (ищем по data-phase внутри текущего блока)
        const phaseEl = genContainer.querySelector(`[data-phase="${data.phase_id}"]`);
        if (phaseEl) phaseEl.classList.add('active');
    }

    onGenerationProgress(data) {
        if (!this.currentExecutionContent) return;
        const progressBar = this.currentExecutionContent.querySelector('.gen-log-progress-fill');
        if (progressBar) {
            let phaseIndex = 0;
            if (data.phase_id === 'graph_algo') phaseIndex = 1;
            if (data.phase_id === 'llm_refine') phaseIndex = 2;
            
            const totalProgress = (phaseIndex * 33.3) + (data.progress * 0.333);
            progressBar.style.width = `${totalProgress}%`;
        }
    }

    onGenerationPhaseCompleted(data) {
        if (!this.currentExecutionContent) return;
        const phaseEl = this.currentExecutionContent.querySelector(`[data-phase="${data.phase_id}"]`);
        if (phaseEl) {
            phaseEl.classList.remove('active');
            phaseEl.classList.add('completed');
            const status = phaseEl.querySelector('.gen-status');
            if (status) status.textContent = '[OK]';
        }
        
        // Показываем content в логе (если есть)
        if (data.content) {
            this.addLog('info', `[INFO] ${data.content}`);
        }
        
        // NLI: показать структуру коннекторов
        if (data.metadata && data.metadata.connector_chain) {
            this.addLog('info', `[NLI] ${data.metadata.connector_chain}`);
        }
        
        // Если это последняя фаза, форсируем 100% на прогресс-баре
        if (data.phase_id === 'llm_refine') {
            const progressBar = this.currentExecutionContent.querySelector('.gen-log-progress-fill');
            if (progressBar) progressBar.style.width = '100%';
        }
    }
    
    onStepStarted(data) {
        this.currentStepIndex = data.stepIndex;
        this.updateStepStatus(data.stepIndex, 'in-progress');
        
        const stepNameEl = document.getElementById('current-step-name');
        if (stepNameEl) stepNameEl.textContent = data.stepName || '';
        
        const totalSteps = (this.currentWorkflow && this.currentWorkflow.steps) 
            ? this.currentWorkflow.steps.length : '?';
        const statEl = document.getElementById('stat-current-step');
        if (statEl) statEl.textContent = `${data.stepIndex + 1}/${totalSteps}`;
        
        // Очищаем панель соревнования
        const competingEl = document.getElementById('competing-agents');
        if (competingEl) competingEl.innerHTML = '';
        
        // Добавляем кандидатов
        if (data.candidateAgents && data.candidateAgents.length > 0) {
            data.candidateAgents.forEach(agentId => {
                const agent = this.agentsLibrary[agentId];
                if (agent) {
                    this.addCompetingAgent(agent);
                } else {
                    // Агент не в библиотеке — создаем временную карточку
                    this.addCompetingAgent({
                        id: agentId,
                        name: agentId,
                        icon: 'T',
                        color: '#4c6ef5',
                        cost: 0,
                        metrics: { avgScore: 0.8, avgResponseTime: 2000 }
                    });
                }
            });
        }
        
        this.addLog('info', `Шаг ${data.stepIndex + 1}: ${data.stepName}`);
    }
    
    onAgentProgress(data) {
        const agentCard = document.querySelector(`[data-agent-id="${data.agentId}"]`);
        if (agentCard && !agentCard.classList.contains('winner')) {
            agentCard.querySelector('.progress-bar').style.width = `${data.progress}%`;
            agentCard.querySelector('.progress-text').textContent = `${data.progress}%`;
        }
    }
    
    onAgentScoreUpdated(data) {
        // Обновляем scores всех агентов из массива
        if (data.agents && Array.isArray(data.agents)) {
            data.agents.forEach(agentData => {
                const agentCard = document.querySelector(`[data-agent-id="${agentData.agentId}"]`);
                if (agentCard) {
                    const scoreValue = agentCard.querySelector('.metric-value.score');
                    if (scoreValue) scoreValue.textContent = (agentData.score * 100).toFixed(0) + '%';
                }
            });
            this.updateLeader(data.agents);
        } else if (data.agentId) {
            // Fallback для одиночного обновления
            const agentCard = document.querySelector(`[data-agent-id="${data.agentId}"]`);
            if (agentCard) {
                const scoreValue = agentCard.querySelector('.metric-value.score');
                if (scoreValue) scoreValue.textContent = (data.score * 100).toFixed(0) + '%';
            }
        }
    }
    
    updateLeader(agents) {
        document.querySelectorAll('.agent-competing-card').forEach(card => {
            if (!card.classList.contains('winner')) {
                card.classList.remove('leading');
            }
        });
        let leader = null;
        let maxScore = -1;
        agents.forEach(agent => {
            if (agent.score !== null && agent.score > maxScore) {
                maxScore = agent.score;
                leader = agent.agentId;
            }
        });
        if (leader) {
            const leaderCard = document.querySelector(`[data-agent-id="${leader}"]`);
            if (leaderCard && !leaderCard.classList.contains('winner')) {
                leaderCard.classList.add('leading');
            }
        }
    }
    
    onAgentSelected(data) {
        const winner = this.agentsLibrary[data.winnerId] || { name: data.winnerId };
        this.addLog('success', `[SELECTED] ${winner.name} (${(data.score * 100).toFixed(0)}%)`);
        
        document.querySelectorAll('.agent-competing-card').forEach(card => {
            const cardId = card.getAttribute('data-agent-id');
            card.classList.remove('competing', 'leading');
            
            if (cardId === data.winnerId) {
                card.classList.add('winner');
                const badge = card.querySelector('.agent-status-badge');
                if (badge) {
                    badge.textContent = 'Executing';
                    badge.className = 'agent-status-badge winner';
                }
                // СБРОС ПРОГРЕССА ДЛЯ "ВТОРОГО ПРОХОДА"
                const progressBar = card.querySelector('.progress-bar');
                const progressText = card.querySelector('.progress-text');
                if (progressBar) progressBar.style.width = '0%';
                if (progressText) progressText.textContent = '0%';
            } else {
                card.classList.add('eliminated');
                const badge = card.querySelector('.agent-status-badge');
                if (badge) badge.textContent = 'Eliminated';
            }
        });
        
        // Удаляем проигравших через 1.5 секунды
        setTimeout(() => {
            document.querySelectorAll('.agent-competing-card.eliminated').forEach(card => {
                card.style.display = 'none';
            });
        }, 1500);
    }
    
    onAgentExecuting(data) {
        const agentCard = document.querySelector(`[data-agent-id="${data.agentId}"]`);
        if (agentCard && agentCard.classList.contains('winner')) {
            const progressBar = agentCard.querySelector('.progress-bar');
            const progressText = agentCard.querySelector('.progress-text');
            
            if (progressBar) progressBar.style.width = `${data.progress}%`;
            if (progressText) progressText.textContent = `${data.progress}%`;
            
            let actionLog = agentCard.querySelector('.agent-action-log');
            if (!actionLog) {
                actionLog = document.createElement('div');
                actionLog.className = 'agent-action-log';
                actionLog.style.fontSize = '11px';
                actionLog.style.color = '#4c6ef5';
                actionLog.style.marginTop = '8px';
                actionLog.style.fontWeight = '600';
                agentCard.querySelector('.agent-progress').appendChild(actionLog);
            }
            actionLog.textContent = `▶ ${data.action}`;
        }
        
        if (data.progress === 100) {
            const agentInfo = this.agentsLibrary[data.agentId] || { name: data.agentId };
            this.addLog('success', `[DONE] ${agentInfo.name}: ${data.action}`);
        }
    }
    
    onStepCompleted(data) {
        this.updateStepStatus(this.currentStepIndex, 'completed');
        
        const totalSteps = (this.currentWorkflow && this.currentWorkflow.steps) 
            ? this.currentWorkflow.steps.length : 1;
        const progress = Math.round(((this.currentStepIndex + 1) / totalSteps) * 100);
        
        const progressEl = document.getElementById('stat-progress');
        if (progressEl) progressEl.textContent = `${progress}%`;
        
        const progressFillEl = document.getElementById('stat-progress-fill');
        if (progressFillEl) progressFillEl.style.width = `${progress}%`;
    }
    
    onWorkflowCompleted(data) {
        console.log('[SUCCESS] Workflow completed');
        this.addLog('success', '[SUCCESS] Workflow успешно завершен!');
        
        this.stopTimer();
        this.isWorkflowFinished = true;
        this.setStopState(false);
        
        if (data.finalAnswer) {
            setTimeout(() => {
                this.addAssistantMessage(data.finalAnswer, "Графовый Архитектор");
            }, 500);
        }
        
        setTimeout(() => {
            if (this.currentExecutionBlock) {
                // НЕ сворачиваем блок, чтобы детали были видны
                // this.currentExecutionBlock.classList.add('collapsed');
                const title = this.currentExecutionBlock.querySelector('.exec-title');
                const icon = this.currentExecutionBlock.querySelector('.toggle-icon');
                if (title) title.textContent = 'Детали выполнения графа';
                if (icon) icon.textContent = '▼';
            }
        }, 1500);
        
        document.getElementById('stat-status').textContent = 'Finished';
    }
    
    onWorkflowStopped() {
        this.addLog('warning', '⏹ Граф остановлен');
        this.resetUI();
    }
    
    onWorkflowError(data) {
        this.addLog('error', `X Ошибка: ${data.error}`);
        this.resetUI();
    }
    
    // === UI Rendering ===
    
    renderWorkflowSteps(steps) {
        const container = document.getElementById('workflow-steps');
        container.innerHTML = '';
        steps.forEach((step, index) => {
            const stepItem = document.createElement('div');
            stepItem.className = 'step-item';
            stepItem.setAttribute('data-step-index', index);
            stepItem.innerHTML = `<div class="step-card pending"><div class="step-name">${step.name}</div><div class="step-status">...</div></div>`;
            container.appendChild(stepItem);
            if (index < steps.length - 1) {
                const arrow = document.createElement('div');
                arrow.className = 'step-arrow';
                arrow.textContent = '→';
                container.appendChild(arrow);
            }
        });
    }
    
    updateStepStatus(stepIndex, status) {
        const stepItem = document.querySelector(`[data-step-index="${stepIndex}"]`);
        if (!stepItem) return;
        const card = stepItem.querySelector('.step-card');
        card.classList.remove('pending', 'in-progress', 'completed');
        card.classList.add(status);
        stepItem.querySelector('.step-status').textContent = status === 'in-progress' ? '>>>' : (status === 'completed' ? '[OK]' : '...');
    }
    
    addCompetingAgent(agent) {
        const container = document.getElementById('competing-agents');
        const card = document.createElement('div');
        card.className = 'agent-competing-card competing';
        card.setAttribute('data-agent-id', agent.id);
        const cost = agent.cost || 0;
        const quality = Math.round((agent.metrics?.avgScore || 0) * 100);
        const time = agent.metrics?.avgResponseTime || 0;
        card.innerHTML = `
            <div class="agent-card-header">
                <div class="agent-avatar" style="background: ${agent.color}20; color: ${agent.color}">${agent.icon}</div>
                <div class="agent-info-text">
                    <h4>${agent.name}</h4>
                    <div class="agent-badges-row">
                        <span class="badge-cost">$${cost.toFixed(3)}</span>
                        <span class="badge-quality">🏆 ${quality}%</span>
                        <span class="badge-time">${time}ms</span>
                    </div>
                </div>
                <div class="agent-status-badge competing">Competing</div>
            </div>
            <div class="agent-progress">
                <div class="progress-bar-container"><div class="progress-bar" style="width: 0%"></div></div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 10px; color: #868e96;">Progress</span>
                    <span class="progress-text" style="font-size: 10px; font-weight: 700; color: #4c6ef5;">0%</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    }
    
    renderAgentLibrary(agents) {
        const container = document.getElementById('agent-library');
        container.innerHTML = '';
        agents.forEach(agent => {
            const card = document.createElement('div');
            card.className = 'library-agent-card';
            const cost = agent.cost || 0;
            const quality = Math.round((agent.metrics?.avgScore || 0) * 100);
            card.innerHTML = `
                <div class="library-agent-avatar" style="background: ${agent.color}20; color: ${agent.color}">${agent.icon}</div>
                <div class="library-agent-info">
                    <h5>${agent.name}</h5>
                    <div class="agent-badges-row">
                        <span class="badge-cost">$${cost.toFixed(3)}</span>
                        <span class="badge-quality">🏆 ${quality}%</span>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    }
    
    filterAgentLibrary(term) {
        document.querySelectorAll('.library-agent-card').forEach(card => {
            const name = card.querySelector('h5').textContent.toLowerCase();
            card.style.display = name.includes(term.toLowerCase()) ? 'flex' : 'none';
        });
    }
    
    startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const s = Math.floor(elapsed / 1000);
            document.getElementById('stat-time').textContent = `${Math.floor(s/60)}:${(s%60).toString().padStart(2,'0')}`;
        }, 1000);
    }
    
    stopTimer() { clearInterval(this.timerInterval); }
    
    resetUI() {
        this.setStopState(false);
        document.getElementById('stat-status').textContent = 'Готов';
        document.getElementById('stat-progress').textContent = '0%';
        document.getElementById('stat-progress-fill').style.width = '0%';
        document.getElementById('competing-agents').innerHTML = '<div class="no-agents-message">Агенты появятся при работе шага</div>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.workflowApp = new WorkflowVisualizer();
});
