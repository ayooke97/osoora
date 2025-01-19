// Configuration object for API
const config = {
    proxyUrl: 'http://localhost:3000/api/chat'
};

// Create axios instance with default config
const api = axios.create({
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
});

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const connectionStatus = document.getElementById('connection-status');
const historyList = document.getElementById('history-list');

// Log DOM elements to console
console.log('DOM Elements loaded:', {
    chatMessages,
    userInput,
    sendButton,
    connectionStatus,
    historyList
});

// Chat history data
let conversations = [];
let currentConversationId = null;

// Function to generate a unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Function to create a new conversation
function createNewConversation() {
    const conversation = {
        id: generateId(),
        messages: [],
        timestamp: new Date(),
        preview: ''
    };
    conversations.unshift(conversation);
    currentConversationId = conversation.id;
    saveConversations();
    updateHistoryList();
    return conversation;
}

// Function to save conversations to localStorage
function saveConversations() {
    localStorage.setItem('chatConversations', JSON.stringify(conversations));
}

// Function to load conversations from localStorage
function loadConversations() {
    const saved = localStorage.getItem('chatConversations');
    if (saved) {
        conversations = JSON.parse(saved);
        updateHistoryList();
    }
}

// Function to update the history list UI
function updateHistoryList() {
    historyList.innerHTML = '';
    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'history-item';
        if (conv.id === currentConversationId) {
            item.classList.add('active');
        }
        
        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date(conv.timestamp).toLocaleString();
        
        const preview = document.createElement('div');
        preview.className = 'preview';
        preview.textContent = conv.preview || 'Empty conversation';
        
        item.appendChild(timestamp);
        item.appendChild(preview);
        
        item.addEventListener('click', () => loadConversation(conv.id));
        historyList.appendChild(item);
    });
}

// Function to load a specific conversation
function loadConversation(conversationId) {
    currentConversationId = conversationId;
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
        chatMessages.innerHTML = '';
        conversation.messages.forEach(msg => {
            addMessage(msg.content, msg.role === 'user', false);
        });
        updateHistoryList();
    }
}

// Function to update connection status
function updateConnectionStatus(message, type = '') {
    console.log('Status Update:', message, type);
    connectionStatus.textContent = message;
    connectionStatus.className = 'connection-status ' + type;
}

// Function to create a message element
function createMessageElement(content, isUser) {
    console.log(`Creating ${isUser ? 'user' : 'bot'} message:`, content);
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    // Create avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser ? '<i class="ri-user-line"></i>' : '<i class="ri-robot-line"></i>';
    
    // Create content wrapper
    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'message-content-wrapper';
    
    // Create sender name
    const sender = document.createElement('div');
    sender.className = 'message-sender';
    sender.textContent = isUser ? 'You' : 'Dashscope AI';
    
    // Create message content
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (isUser) {
        messageContent.textContent = content;
    } else {
        messageContent.innerHTML = marked.parse(content, {
            gfm: true,
            breaks: true,
            highlight: function(code, language) {
                if (language && hljs.getLanguage(language)) {
                    try {
                        return hljs.highlight(code, { language }).value;
                    } catch (err) {
                        console.error('Highlight error:', err);
                    }
                }
                return code;
            }
        });
        
        // Make links open in new tab
        messageContent.querySelectorAll('a').forEach(link => {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        });
    }
    
    contentWrapper.appendChild(sender);
    contentWrapper.appendChild(messageContent);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentWrapper);
    return messageDiv;
}

// Function to add a message to the chat
function addMessage(content, isUser, save = true) {
    const messageElement = createMessageElement(content, isUser);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (save) {
        if (!currentConversationId) {
            createNewConversation();
        }
        
        const conversation = conversations.find(c => c.id === currentConversationId);
        if (conversation) {
            conversation.messages.push({ role: isUser ? 'user' : 'assistant', content });
            conversation.preview = content;
            conversation.timestamp = new Date();
            saveConversations();
            updateHistoryList();
        }
    }
}

// Function to show typing indicator
function showTypingIndicator() {
    console.log('Showing typing indicator');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="ri-robot-line"></i>';
    
    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        bubble.appendChild(dot);
    }
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(bubble);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to remove typing indicator
function removeTypingIndicator() {
    console.log('Removing typing indicator');
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Function to send message to API
async function sendToAPI(message) {
    try {
        console.log('Sending message to API:', message);
        updateConnectionStatus('Connecting...', '');
        
        console.log('Using proxy URL:', config.proxyUrl);
        
        const response = await api.post(config.proxyUrl, {
            message: message
        });
        
        console.log('API Response:', response.data);
        
        updateConnectionStatus('Connected', 'connected');
        
        if (response.data && response.data.output && response.data.output.text) {
            return {
                choices: [
                    {
                        message: {
                            content: response.data.output.text
                        }
                    }
                ]
            };
        } else {
            console.error('Invalid API response format:', response.data);
            throw new Error('Invalid response format from API');
        }
    } catch (error) {
        console.error('API Error:', error);
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
            console.error('Response headers:', error.response.headers);
        }
        updateConnectionStatus(`Error: ${error.response?.data?.message || error.message}`, 'error');
        throw error;
    }
}

// Function to handle sending messages
async function handleSendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    console.log('Handling new message:', message);
    
    // Clear input
    userInput.value = '';

    // Add user message to chat
    addMessage(message, true);

    // Show typing indicator
    showTypingIndicator();

    try {
        // Send message to API
        const response = await sendToAPI(message);
        
        // Remove typing indicator
        removeTypingIndicator();

        // Add bot response to chat
        if (response && response.choices && response.choices[0].message) {
            const botResponse = response.choices[0].message.content;
            console.log('Bot response:', botResponse);
            addMessage(botResponse, false);
        } else {
            throw new Error('Invalid API response format');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again later.', false);
    }
}

// Event listeners
sendButton.addEventListener('click', handleSendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSendMessage();
    }
});

// Load conversations on startup
loadConversations();

// Initial greeting
addMessage('Hello! I am your Dashscope AI assistant. How can I help you today?', false);
