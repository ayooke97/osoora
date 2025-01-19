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

// Log DOM elements to console
console.log('DOM Elements loaded:', {
    chatMessages,
    userInput,
    sendButton,
    connectionStatus
});

// Function to update connection status
function updateConnectionStatus(message, type = '') {
    console.log('Status Update:', message, type);
    connectionStatus.textContent = message;
    connectionStatus.className = 'connection-status ' + type;
}

// Chat history
let chatHistory = [];

// Function to create a message element
function createMessageElement(content, isUser) {
    console.log(`Creating ${isUser ? 'user' : 'bot'} message:`, content);
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (isUser) {
        messageContent.textContent = content;
    } else {
        // Parse markdown for bot messages
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
    
    messageDiv.appendChild(messageContent);
    return messageDiv;
}

// Function to add a message to the chat
function addMessage(content, isUser) {
    const messageElement = createMessageElement(content, isUser);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatHistory.push({ role: isUser ? 'user' : 'assistant', content });
}

// Function to show typing indicator
function showTypingIndicator() {
    console.log('Showing typing indicator');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';
    
    const typingContent = document.createElement('div');
    typingContent.className = 'typing-indicator';
    typingContent.textContent = 'Bot is typing...';
    
    typingDiv.appendChild(typingContent);
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

// Initial greeting
addMessage('Hello! I am your Dashscope AI assistant. How can I help you today?', false);
