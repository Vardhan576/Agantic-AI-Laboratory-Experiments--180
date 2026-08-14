document.addEventListener("DOMContentLoaded", () => {
    const chatHistoryBox = document.getElementById("chat-history-box");
    const chatInputField = document.getElementById("chat-input-field");
    const chatSendBtn = document.getElementById("chat-send-btn");
    const traceLogsBox = document.getElementById("trace-logs-box");
    const clearTraceBtn = document.getElementById("clear-trace-btn");
    
    // Elements for dashboard metrics
    const metricCustomers = document.getElementById("metric-customers");
    const metricProducts = document.getElementById("metric-products");
    const metricOrders = document.getElementById("metric-orders");
    const metricRevenue = document.getElementById("metric-revenue");

    // Fetch and update dashboard metrics
    async function updateDashboardMetrics() {
        try {
            const response = await fetch("/api/status");
            const data = await response.json();
            if (data.status === "online") {
                metricCustomers.textContent = data.metrics.customers.toLocaleString();
                metricProducts.textContent = data.metrics.products.toLocaleString();
                metricOrders.textContent = data.metrics.orders.toLocaleString();
                metricRevenue.textContent = `$${data.metrics.revenue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
        } catch (error) {
            console.error("Error updating dashboard metrics:", error);
        }
    }

    // Call metrics update on load
    updateDashboardMetrics();

    // Clear trace logs
    clearTraceBtn.addEventListener("click", () => {
        traceLogsBox.innerHTML = `
            <div class="trace-placeholder">
                <p>Waiting for query execution... Trace thoughts, SQL commands, and RAG contexts will appear here in real-time.</p>
            </div>
        `;
    });

    // Helper to add chat messages to UI
    function addChatMessage(sender, text, avatarIcon) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message");
        if (sender === "user") {
            messageDiv.classList.add("user-message");
        }
        
        messageDiv.innerHTML = `
            <div class="avatar">${avatarIcon}</div>
            <div class="message-content">
                <p>${text.replace(/\n/g, "<br>")}</p>
            </div>
        `;
        chatHistoryBox.appendChild(messageDiv);
        chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
    }

    // Render agent trace logs in trace panel
    function renderTraceLogs(logs) {
        traceLogsBox.innerHTML = ""; // Clear placeholder
        
        logs.forEach((logLine, idx) => {
            setTimeout(() => {
                const logElement = document.createElement("div");
                logElement.classList.add("trace-line");
                
                // Color coding based on tags
                if (logLine.includes("[CENSORED]") || logLine.includes("Error")) {
                    logElement.classList.add("warning");
                } else if (logLine.includes("[DB Agent]") || logLine.includes("[RAG Agent]")) {
                    logElement.classList.add("success");
                } else if (logLine.includes("[Triage]") || logLine.includes("[Supervisor]")) {
                    logElement.classList.add("info");
                }
                
                logElement.textContent = logLine;
                traceLogsBox.appendChild(logElement);
                traceLogsBox.scrollTop = traceLogsBox.scrollHeight;
            }, idx * 250); // Small cascade delay
        });
    }

    // Send chat function
    async function sendMessage() {
        const messageText = chatInputField.value.trim();
        if (!messageText) return;

        // 1. Add user message
        addChatMessage("user", messageText, "👤");
        chatInputField.value = "";

        // 2. Add loading message for agent
        const loadingDiv = document.createElement("div");
        loadingDiv.classList.add("message", "loading-message");
        loadingDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                <p>Thinking...</p>
            </div>
        `;
        chatHistoryBox.appendChild(loadingDiv);
        chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;

        try {
            // 3. Post to API
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: messageText })
            });

            // Remove loading
            loadingDiv.remove();

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server error occurred");
            }

            const data = await response.json();
            
            // 4. Render logs and final agent response
            renderTraceLogs(data.trace_logs);
            addChatMessage("agent", data.final_response, "🤖");

            // 5. Update metrics (in case user updated data or queries shifted stats)
            updateDashboardMetrics();

        } catch (error) {
            loadingDiv.remove();
            addChatMessage("agent", `Error: ${error.message}`, "⚠️");
            console.error("Chat error:", error);
        }
    }

    // Trigger events
    chatSendBtn.addEventListener("click", sendMessage);
    chatInputField.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
