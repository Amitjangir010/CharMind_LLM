// ======================================================================================
// script.js
// Refactored, modular, and state-driven JavaScript for the CharMind LLM UI
// ======================================================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    const state = {
        currentMode: 'chat', // 'chat' or 'rlhf'
        isLoading: false,
        isTraining: false,
        chatHistory: [],
        availableModels: [],
        activeModel: '',
        abortController: null,
    };

    // --- DOM Element Cache ---
    const dom = {
        // sidebar
        modelSelect: document.getElementById('model-select'),
        normalModeBtn: document.getElementById('normal-mode-btn'),
        rlhfModeBtn: document.getElementById('rlhf-mode-btn'),
        statusIndicator: document.getElementById('status-indicator'),
        // training controls
        trainingForm: document.getElementById('training-form'),
        trainingModeSelect: document.getElementById('training-mode'),
        runNameInput: document.getElementById('run-name'),
        baseModelSelectorContainer: document.getElementById('base-model-selector-container'),
        baseModelSelect: document.getElementById('base-model-select'),
        baseModelLabel: document.querySelector('label[for="base-model-select"]'), // Added this line
        startTrainingBtn: document.getElementById('start-training-btn'),
        // main content
        chatArea: document.getElementById('chat-area'),
        rlhfArea: document.getElementById('rlhf-area'),
        messages: document.getElementById('messages'),
        chatForm: document.getElementById('chat-form'),
        userInput: document.getElementById('user-input'),
        sendBtn: document.getElementById('send-btn'),
        stopBtn: document.getElementById('stop-btn'),
        // rlhf
        rlhfPromptDisplay: document.getElementById('rlhf-prompt-display'),
        rlhfRepliesContainer: document.querySelector('.rlhf-replies'),
        nextPromptBtn: document.getElementById('next-prompt-btn'),
        // monitoring
        trainingStatusArea: document.getElementById('training-status-area'),
        trainingStatusContent: document.getElementById('training-status-content'),
    };

    // --- API Calls ---
    const api = {
        getStatus: async () => (await fetch('/api/status')).json(),
        getModels: async () => (await fetch('/api/available_models')).json(),
        setModel: async (modelId) => fetch('/api/set_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId }),
        }),
        getChatStream: async (history, signal) => {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history }),
                signal,
            });
            if (!response.ok) throw new Error(await response.text());
            return response;
        },
        generateRlhfResponses: async (prompt, signal) => (await fetch('/api/rlhf_generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt }),
            signal,
        })).json(),
        saveFeedback: async (prompt, chosen, rejected) => fetch('/api/save_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, chosen, rejected }),
        }),
        startTraining: async (mode, runName, baseModelPath) => (await fetch('/api/start_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, run_name: runName, base_model_path: baseModelPath }),
        })).json(),
        getTrainingStatus: async () => (await fetch('/api/training_status')).json(),
    };

    // --- UI Rendering ---
    const ui = {
        renderModels: (models, activeModelId) => {
            const selects = [dom.modelSelect, dom.baseModelSelect];
            selects.forEach(select => {
                select.innerHTML = '';
                models.forEach(m => {
                    const option = document.createElement('option');
                    option.value = m.id;
                    option.textContent = m.name;
                    select.appendChild(option);
                });
            });
            if (activeModelId) {
                dom.modelSelect.value = activeModelId;
            }
        },
        addMessage: (role, text, isSpecial = false) => {
            const div = document.createElement('div');
            // Use new message classes: 'message user-message' or 'message bot-message'
            div.className = `message ${role}-message`;
            if (isSpecial) {
                div.classList.add('system-highlight-message');
            }
            
            // Create avatar element - Map assistant to bot-avatar
            const avatar = document.createElement('div');
            let avatarClass = `${role}-avatar`;

            // Map assistant role to bot-avatar class
            if (role === 'assistant') {
                avatarClass = 'bot-avatar';
            }

            avatar.className = `avatar ${avatarClass}`;
            
            // Add content to avatars - UNCOMMENT AND MODIFY THESE LINES
            if (role === 'user') {
                // User avatar will get icon from CSS ::before
            } else if (role === 'assistant') {
                // Bot avatar will get icon from CSS ::before  
            } else if (role === 'system') {
                // System avatar will get icon from CSS ::before
            }

            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = text;

            // Append avatar and message content based on role for proper ordering
            if (role === 'user') {
                div.appendChild(messageContent);
                div.appendChild(avatar);
            } else {
                div.appendChild(avatar);
                div.appendChild(messageContent);
            }

            dom.messages.appendChild(div);
            dom.messages.scrollTop = dom.messages.scrollHeight;
            return messageContent; // Return the content element for streaming updates
        },
        addSystemMessage: (text) => {
            ui.addMessage('system', text, true);
        },
        renderRlhfChoices: (prompt, res) => {
            dom.rlhfPromptDisplay.textContent = `Prompt: "${prompt}"`;
            dom.rlhfRepliesContainer.innerHTML = '';
            const createChoiceBlock = (text, isChoice1) => {
                const block = document.createElement('div');
                block.className = 'rlhf-reply-block';
                block.textContent = text;
                block.onclick = async () => {
                    dom.rlhfRepliesContainer.querySelectorAll('.rlhf-reply-block').forEach(b => {
                        b.onclick = null;
                        if (b !== block) b.style.opacity = '0.5';
                    });
                    block.style.borderColor = 'var(--accent-green)';
                    const chosen = text;
                    const rejected = isChoice1 ? res.response2 : res.response1;
                    await api.saveFeedback(prompt, chosen, rejected);
                    dom.nextPromptBtn.style.display = 'block';
                };
                return block;
            };
            if (res && res.response1 && res.response2) {
                dom.rlhfRepliesContainer.appendChild(createChoiceBlock(res.response1, true));
                dom.rlhfRepliesContainer.appendChild(createChoiceBlock(res.response2, false));
            } else {
                dom.rlhfPromptDisplay.textContent = "Could not generate valid responses.";
            }
        },
        setLoading: (isLoading) => {
            state.isLoading = isLoading;
            dom.sendBtn.style.display = isLoading ? 'none' : 'block';
            dom.stopBtn.style.display = isLoading ? 'block' : 'none';
            dom.statusIndicator.textContent = isLoading ? 'Generating...' : 'Idle';
        },
        toggleMode: (newMode) => {
            state.currentMode = newMode;
            dom.userInput.value = '';
            dom.chatArea.style.display = newMode === 'chat' ? 'flex' : 'none';
            dom.rlhfArea.style.display = newMode === 'rlhf' ? 'block' : 'none';
            dom.normalModeBtn.classList.toggle('active', newMode === 'chat');
            dom.rlhfModeBtn.classList.toggle('active', newMode === 'rlhf');
            dom.userInput.placeholder = newMode === 'chat' ? "Type your message..." : "Enter a prompt for RLHF...";
        },
        renderTrainingStatus: (statuses) => {
            if (Object.keys(statuses).length === 0) {
                dom.trainingStatusContent.innerHTML = '<p>No active or recent training jobs.</p>';
                state.isTraining = false;
                return;
            }
            state.isTraining = Object.values(statuses).some(s => s.status === 'Running');
            dom.startTrainingBtn.disabled = state.isTraining; // Disable button if training is active
            dom.trainingStatusContent.innerHTML = '';
            for (const [name, data] of Object.entries(statuses)) {
                const statusDiv = document.createElement('div');
                statusDiv.className = 'training-job training-card'; // Added training-card class
                
                let statusHtml = ``;
                if (data.status === "Completed") {
                    statusHtml = `<strong class="job-status completed">${data.completion_message}</strong>`;
                } else if (data.status.startsWith("Failed")) {
                    statusHtml = `<strong class="job-status failed">${data.status}</strong>`;
                } else { // Running
                    statusHtml = `
                        <span class="job-status running">${data.status}</span>
                        <span class="progress-percent">${data.progress_percent}%</span>
                        <span class="progress-spinner"></span>
                    `;
                }

                statusDiv.innerHTML = `
                    <div class="job-header">
                        <strong class="job-name">${name}</strong>
                        ${statusHtml}
                    </div>
                `;

                // Add latest loss/reward info
                if (data.latest_train_loss !== null || data.latest_val_loss !== null || data.latest_avg_reward !== null) {
                    let metricsHtml = '<div class="job-metrics">';
                    if (data.latest_train_loss !== null) {
                        metricsHtml += `<span>Train Loss: ${data.latest_train_loss.toFixed(4)}</span>`;
                    }
                    if (data.latest_val_loss !== null) {
                        metricsHtml += `<span>Val Loss: ${data.latest_val_loss.toFixed(4)}</span>`;
                    }
                    if (data.latest_avg_reward !== null) {
                        metricsHtml += `<span>Avg Reward: ${data.latest_avg_reward.toFixed(4)}</span>`;
                    }
                    metricsHtml += '</div>';
                    statusDiv.innerHTML += metricsHtml;
                }

                // Optionally, if you still want some log info, you can add it, but user requested to remove log_preview
                // if (data.log_preview) {
                //     statusDiv.innerHTML += `<pre class="job-log">${data.log_preview}</pre>`;
                // }
                dom.trainingStatusContent.prepend(statusDiv);
            }
        }
    };

    // --- Event Handlers ---
    const handlers = {
        handleFormSubmit: async (e) => {
            e.preventDefault();
            const prompt = dom.userInput.value.trim();
            if (!prompt || state.isLoading) return;
            ui.setLoading(true);
            state.abortController = new AbortController();
            dom.userInput.value = '';

            if (state.currentMode === 'chat') {
                state.chatHistory.push({ role: 'user', content: prompt });
                ui.addMessage('user', prompt);
                const assistantBubble = ui.addMessage('assistant', '...');
                try {
                    const response = await api.getChatStream(state.chatHistory, state.abortController.signal);
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let llmText = '';
                    assistantBubble.textContent = '';
                    while(true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        llmText += decoder.decode(value, { stream: true });
                        assistantBubble.textContent = llmText;
                        dom.messages.scrollTop = dom.messages.scrollHeight;
                    }
                    state.chatHistory.push({ role: 'assistant', content: llmText });
                } catch (error) {
                    if (error.name !== 'AbortError') {
                        assistantBubble.textContent = error.message || 'An error occurred.';
                    } else {
                        assistantBubble.parentElement.remove();
                    }
                    state.chatHistory.pop(); // Remove user message if assistant fails
                }
            } else { // RLHF Mode
                dom.rlhfPromptDisplay.textContent = 'Generating...';
                dom.rlhfRepliesContainer.innerHTML = '';
                try {
                    const responses = await api.generateRlhfResponses(prompt, state.abortController.signal);
                    ui.renderRlhfChoices(prompt, responses);
                } catch (error) {
                     if (error.name !== 'AbortError') {
                        dom.rlhfPromptDisplay.textContent = 'Error generating responses.';
                    }
                }
            }
            ui.setLoading(false);
        },
        handleModelChange: async (e) => {
            const modelId = e.target.value;
            if (modelId === state.activeModel) return;
            await api.setModel(modelId);
            state.activeModel = modelId;
            ui.addSystemMessage(`Model changed to ${e.target.options[e.target.selectedIndex].textContent}`);
        },
        handleStop: () => state.abortController?.abort(),
        handleNextPrompt: () => {
            dom.rlhfPromptDisplay.textContent = '';
            dom.rlhfRepliesContainer.innerHTML = '';
            dom.nextPromptBtn.style.display = 'none';
            dom.userInput.focus();
        },
        handleTrainingModeChange: () => {
            const isRlhfOrFinetune = dom.trainingModeSelect.value === 'rlhf' || dom.trainingModeSelect.value === 'finetune';
            dom.baseModelSelectorContainer.style.display = isRlhfOrFinetune ? 'block' : 'none';
            // Update the label text
            if (dom.baseModelLabel) {
                if (dom.trainingModeSelect.value === 'finetune') {
                    dom.baseModelLabel.textContent = 'Base Model for Fine-tuning';
                } else if (dom.trainingModeSelect.value === 'rlhf') {
                    dom.baseModelLabel.textContent = 'Base Model for RLHF';
                } else {
                    dom.baseModelLabel.textContent = 'Base Model'; // Default or other modes
                }
            }
        },
        handleTrainingSubmit: async (e) => {
            e.preventDefault();
            const mode = dom.trainingModeSelect.value;
            const runName = dom.runNameInput.value.trim();
            const baseModelPath = dom.baseModelSelect.value;
            if (!runName) {
                alert('Please enter a run name.');
                return;
            }
            dom.startTrainingBtn.disabled = true;
            dom.startTrainingBtn.textContent = 'Starting...';
            try {
                const result = await api.startTraining(mode, runName, baseModelPath);
                if (result.status !== 'ok') throw new Error(result.message);
                alert(`Training job '${runName}' started successfully!`);
                dom.runNameInput.value = '';
                await monitor.fetchTrainingStatus();
            } catch (error) {
                alert(`Error starting training: ${error.message}`);
            } finally {
                dom.startTrainingBtn.disabled = false;
                dom.startTrainingBtn.textContent = 'Start Training';
            }
        }
    };

    // --- Monitoring ---
    const monitor = {
        fetchTrainingStatus: async () => {
            try {
                const statuses = await api.getTrainingStatus();
                ui.renderTrainingStatus(statuses);
            } catch (error) {
                console.error("Failed to fetch training status:", error);
            }
        },
        start: () => {
            setInterval(monitor.fetchTrainingStatus, 5000); // Poll every 5 seconds
        }
    };

    // --- Initialization ---
    const init = async () => {
        // Event Listeners
        dom.chatForm.addEventListener('submit', handlers.handleFormSubmit);
        dom.modelSelect.addEventListener('change', handlers.handleModelChange);
        dom.stopBtn.addEventListener('click', handlers.handleStop);
        dom.normalModeBtn.addEventListener('click', () => ui.toggleMode('chat'));
        dom.rlhfModeBtn.addEventListener('click', () => ui.toggleMode('rlhf'));
        dom.nextPromptBtn.addEventListener('click', handlers.handleNextPrompt);
        dom.trainingForm.addEventListener('submit', handlers.handleTrainingSubmit);
        dom.trainingModeSelect.addEventListener('change', handlers.handleTrainingModeChange);

        // Initial state setup
        try {
            const [status, models] = await Promise.all([api.getStatus(), api.getModels()]);
            state.availableModels = models;
            state.activeModel = status.active_model || models[0]?.id;
            ui.renderModels(models, state.activeModel);
            handlers.handleTrainingModeChange();
        } catch (error) {
            console.error("Initialization failed:", error);
            ui.addMessage('system', 'Error connecting to backend. Please refresh.');
        }

        ui.toggleMode('chat');
        monitor.fetchTrainingStatus();
        monitor.start();
    };

    init();
});