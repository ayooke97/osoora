// Session check at the start of the file
function checkSession() {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    if (!token || !user) {
        window.location.href = 'account.html';
        return false;
    }
    return true;
}

// Run session check immediately
if (!checkSession()) {
    // Stop further script execution if not authenticated
    throw new Error('Not authenticated');
}

// Global variables
let conversations = [];
let currentConversationId = null;

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
const newChatButton = document.getElementById('new-chat');
const clearHistoryButton = document.getElementById('clear-history');
const emptyHistory = document.getElementById('empty-history');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const userButton = document.getElementById('user-button');
const userDropdown = document.getElementById('user-dropdown');
const loginButton = document.getElementById('login-button');
const logoutButton = document.getElementById('logout-button');
const userAvatar = document.getElementById('user-avatar');
const userName = document.getElementById('user-name');
const dropdownUserName = document.getElementById('dropdown-user-name');
const userEmail = document.getElementById('user-email');
const menuToggle = document.getElementById('menu-toggle');
const chatHistory = document.getElementById('chat-history');
const closeSidebar = document.getElementById('close-sidebar');

// Log DOM elements to console
console.log('DOM Elements loaded:', {
    chatMessages,
    userInput,
    sendButton,
    connectionStatus,
    historyList,
    newChatButton,
    clearHistoryButton,
    emptyHistory,
    themeToggle,
    themeIcon,
    userButton,
    userDropdown,
    loginButton,
    logoutButton,
    userAvatar,
    userName,
    dropdownUserName,
    userEmail
});

// Function to generate a unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Function to format timestamp
function formatTimestamp(date) {
    const now = new Date();
    const messageDate = new Date(date);
    
    // If today, show time
    if (messageDate.toDateString() === now.toDateString()) {
        return messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If this year, show month and day
    if (messageDate.getFullYear() === now.getFullYear()) {
        return messageDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    // Otherwise show full date
    return messageDate.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}

// Function to create a new conversation
function createNewConversation() {
    const conversation = {
        id: generateId(),
        messages: [],
        timestamp: new Date(),
        preview: '',
        topic: 'New Chat'  // Default topic
    };
    conversations.unshift(conversation);
    currentConversationId = conversation.id;
    saveConversations();
    updateHistoryList();
    clearChat();

    // Add welcome message
    const welcomeMessage = "👋 Hello! I'm your AI assistant. How can I help you today?";
    addMessage(welcomeMessage, false);
    conversation.messages.push({ role: 'assistant', content: welcomeMessage });
    conversation.preview = welcomeMessage;
    saveConversations();
    updateHistoryList();

    return conversation;
}

// Function to generate topic from messages
function generateTopic(messages) {
    if (messages.length === 0) return 'New Chat';
    // Get the first user message as the topic
    const firstMessage = messages.find(m => m.role === 'user');
    if (!firstMessage) return 'New Chat';
    
    // Truncate the message to create a topic
    const topic = firstMessage.content.split('\n')[0].trim();
    return topic.length > 40 ? topic.substring(0, 37) + '...' : topic;
}

// Function to update conversation topic
function updateConversationTopic(conversationId) {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
        conversation.topic = generateTopic(conversation.messages);
        saveConversations();
        updateHistoryList();
    }
}

// Function to clear chat messages
function clearChat() {
    chatMessages.innerHTML = '';
    userInput.value = '';
    userInput.focus();
}

// Function to save conversations to localStorage
function saveConversations() {
    localStorage.setItem('conversations', JSON.stringify(conversations));
    toggleEmptyState();
}

// Function to load conversations from localStorage
function loadConversations() {
    const savedConversations = localStorage.getItem('conversations');
    if (savedConversations) {
        conversations = JSON.parse(savedConversations);
        if (conversations.length > 0) {
            currentConversationId = conversations[0].id;
            // Convert stored timestamps back to Date objects
            conversations.forEach(conv => {
                conv.timestamp = new Date(conv.timestamp);
            });
        }
    }
    updateHistoryList();
}

// Function to toggle empty state
function toggleEmptyState() {
    if (conversations.length === 0) {
        emptyHistory.style.display = 'flex';
    } else {
        emptyHistory.style.display = 'none';
    }
}

// Function to delete conversation
function deleteConversation(id, event) {
    event.stopPropagation();
    if (confirm('Are you sure you want to delete this conversation?')) {
        conversations = conversations.filter(c => c.id !== id);
        if (id === currentConversationId) {
            currentConversationId = conversations.length > 0 ? conversations[0].id : null;
            if (currentConversationId) {
                loadConversation(currentConversationId);
            } else {
                clearChat();
            }
        }
        saveConversations();
        updateHistoryList();
    }
}

// Function to update the history list UI
function updateHistoryList() {
    historyList.innerHTML = '';
    
    if (conversations.length === 0) {
        historyList.appendChild(emptyHistory);
        emptyHistory.style.display = 'flex';
    } else {
        emptyHistory.style.display = 'none';
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'history-item';
            if (conv.id === currentConversationId) {
                item.classList.add('active');
            }
            
            const header = document.createElement('div');
            header.className = 'history-item-header';
            
            const topic = document.createElement('div');
            topic.className = 'topic';
            topic.textContent = conv.topic || 'New Chat';
            
            const timestamp = document.createElement('div');
            timestamp.className = 'timestamp';
            timestamp.textContent = formatTimestamp(conv.timestamp);
            
            const preview = document.createElement('div');
            preview.className = 'preview';
            preview.textContent = conv.preview || 'Empty conversation';
            
            const actions = document.createElement('div');
            actions.className = 'actions';
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.innerHTML = '<i class="ri-delete-bin-line"></i>';
            deleteBtn.title = 'Delete conversation';
            deleteBtn.onclick = (e) => deleteConversation(conv.id, e);
            
            header.appendChild(topic);
            header.appendChild(deleteBtn);
            
            item.appendChild(header);
            item.appendChild(timestamp);
            item.appendChild(preview);
            
            item.addEventListener('click', () => loadConversation(conv.id));
            historyList.appendChild(item);
        });
    }
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

// Configure marked with options
marked.setOptions({
    highlight: function(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    },
    langPrefix: 'hljs language-',
    gfm: true,
    breaks: true
});

// Function to create a message element
function createMessageElement(content, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.innerHTML = isUser ? '<i class="ri-user-line"></i>' : '<i class="ri-robot-line"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isUser) {
        contentDiv.textContent = content;
    } else {
        // For bot messages, render as Markdown
        contentDiv.innerHTML = marked.parse(content);
        // Highlight any code blocks
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    return { messageDiv, contentDiv };
}

// Function to add a message to the chat
function addMessage(content, isUser, save = true) {
    const { messageDiv, contentDiv } = createMessageElement(content, isUser);
    chatMessages.appendChild(messageDiv);
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
            // Update topic after adding first user message
            if (isUser && conversation.messages.filter(m => m.role === 'user').length === 1) {
                updateConversationTopic(currentConversationId);
            }
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
        
        // Create event source for streaming
        const response = await fetch(config.proxyUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        // Create a reader to read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        updateConnectionStatus('Connected', 'connected');

        // Return a promise that resolves with an array of all responses
        return new Promise(async (resolve, reject) => {
            try {
                let fullResponse = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        break;
                    }

                    // Decode the chunk and add to buffer
                    buffer += decoder.decode(value, { stream: true });
                    
                    // Process complete events
                    let newlineIndex;
                    while ((newlineIndex = buffer.indexOf('\n\n')) !== -1) {
                        const event = buffer.slice(0, newlineIndex);
                        buffer = buffer.slice(newlineIndex + 2);

                        // Parse the event
                        const lines = event.split('\n');
                        for (const line of lines) {
                            if (!line.startsWith('data:')) continue;
                            
                            const data = line.slice(5).trim();
                            if (!data) continue;
                            
                            if (data === '[DONE]') {
                                resolve({
                                    choices: [{
                                        message: {
                                            content: fullResponse
                                        }
                                    }]
                                });
                                return;
                            }

                            try {
                                const parsedData = JSON.parse(data);
                                if (parsedData.error) {
                                    throw new Error(parsedData.error);
                                }
                                if (parsedData.choices?.[0]?.message?.content) {
                                    const content = parsedData.choices[0].message.content;
                                    if (content.trim()) {
                                        fullResponse += content;
                                        // Emit partial response for real-time updates
                                        window.dispatchEvent(new CustomEvent('partialResponse', {
                                            detail: { content }
                                        }));
                                    }
                                }
                            } catch (e) {
                                if (!data.startsWith('id:')) {  // Ignore ID lines
                                    console.error('Error parsing SSE data:', e, 'Raw data:', data);
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                reject(error);
            }
        });
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
        // Remove typing indicator before creating new message
        removeTypingIndicator();

        // Create a temporary message element for streaming updates
        const tempMessageId = 'temp-' + Date.now();
        const { messageDiv, contentDiv } = createMessageElement('', false);
        contentDiv.id = tempMessageId;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        let accumulatedMarkdown = '';

        // Listen for partial responses
        const handlePartialResponse = (event) => {
            const content = event.detail.content;
            accumulatedMarkdown += content;
            const messageElement = document.getElementById(tempMessageId);
            if (messageElement) {
                messageElement.innerHTML = marked.parse(accumulatedMarkdown);
                // Highlight any code blocks
                messageElement.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightBlock(block);
                });
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        };
        window.addEventListener('partialResponse', handlePartialResponse);

        // Send message to API
        const response = await sendToAPI(message);
        
        // Remove event listener
        window.removeEventListener('partialResponse', handlePartialResponse);

        // Update the final message
        if (response && response.choices && response.choices[0].message) {
            const botResponse = response.choices[0].message.content;
            console.log('Bot response:', botResponse);
            const messageElement = document.getElementById(tempMessageId);
            if (messageElement) {
                messageElement.innerHTML = marked.parse(botResponse);
                // Highlight any code blocks
                messageElement.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightBlock(block);
                });
            }
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Save the bot's response to conversation history
            const conversation = conversations.find(c => c.id === currentConversationId);
            if (conversation) {
                conversation.messages.push({ role: 'assistant', content: botResponse });
                conversation.preview = botResponse;
                conversation.timestamp = new Date();
                saveConversations();
                updateHistoryList();
            }
        } else {
            throw new Error('Invalid API response format');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again later.', false);
    }
}

// Function to clear all history
function clearAllHistory() {
    if (confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
        conversations = [];
        currentConversationId = null;
        localStorage.removeItem('conversations');
        clearChat();
        updateHistoryList();
        toggleEmptyState();
        // Create a new conversation after clearing
        createNewConversation();
    }
}

// Theme management
function getPreferredTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        return savedTheme;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function updateThemeIcon(theme) {
    const isDark = theme === 'dark';
    themeIcon.className = isDark ? 'ri-sun-line' : 'ri-moon-line';
    themeToggle.setAttribute('title', `Switch to ${isDark ? 'light' : 'dark'} mode`);
    themeToggle.setAttribute('aria-label', `Switch to ${isDark ? 'light' : 'dark'} mode`);
}

function setTheme(theme, updateStorage = true) {
    document.documentElement.style.setProperty('--transition-normal', 'none');
    document.body.classList.remove('light-theme', 'dark-theme');
    
    requestAnimationFrame(() => {
        document.body.classList.add(`${theme}-theme`);
        document.documentElement.style.setProperty('--transition-normal', 'all 0.3s ease');
        
        if (updateStorage) {
            localStorage.setItem('theme', theme);
        }
        
        updateThemeIcon(theme);
    });
}

// Initialize theme
const initialTheme = getPreferredTheme();
setTheme(initialTheme, false);

// Theme toggle event listener
themeToggle.addEventListener('click', () => {
    const currentTheme = localStorage.getItem('theme') || getPreferredTheme();
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
});

// Watch for system theme changes
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
mediaQuery.addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light', false);
    }
});

// Mobile menu toggle
menuToggle.addEventListener('click', () => {
    chatHistory.classList.toggle('show');
});

// Close sidebar button
closeSidebar.addEventListener('click', () => {
    chatHistory.classList.remove('show');
});

// Close menu when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768) {
        const isClickInsideHistory = chatHistory.contains(e.target);
        const isClickOnToggle = menuToggle.contains(e.target);
        
        if (!isClickInsideHistory && !isClickOnToggle && chatHistory.classList.contains('show')) {
            chatHistory.classList.remove('show');
        }
    }
});

// User Account Management
userButton.addEventListener('click', () => {
    userDropdown.classList.toggle('hidden');
});

document.addEventListener('click', (e) => {
    if (!userButton.contains(e.target) && !userDropdown.contains(e.target)) {
        userDropdown.classList.add('hidden');
    }
});

loginButton.addEventListener('click', () => {
    window.location.href = 'account.html';
});

logoutButton.addEventListener('click', () => {
    if (confirm('Are you sure you want to sign out?')) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        updateUserInterface();
    }
});

function updateUserInterface() {
    const user = JSON.parse(localStorage.getItem('user'));
    const isAuthenticated = !!localStorage.getItem('token');

    if (isAuthenticated && user) {
        // Update avatar
        const avatarUrl = user.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(user.name)}&background=7c3aed&color=fff`;
        userAvatar.src = avatarUrl;
        document.querySelector('.user-info img').src = avatarUrl;

        // Update name and email
        userName.textContent = user.name;
        dropdownUserName.textContent = user.name;
        userEmail.textContent = user.email;

        // Show/hide buttons
        loginButton.classList.add('hidden');
        logoutButton.classList.remove('hidden');
    } else {
        // Reset to guest state
        const guestAvatar = 'https://ui-avatars.com/api/?name=Guest&background=7c3aed&color=fff';
        userAvatar.src = guestAvatar;
        document.querySelector('.user-info img').src = guestAvatar;

        userName.textContent = 'Guest';
        dropdownUserName.textContent = 'Guest';
        userEmail.textContent = 'Not signed in';

        loginButton.classList.remove('hidden');
        logoutButton.classList.add('hidden');
    }
}

updateUserInterface();

// Event listeners
sendButton.addEventListener('click', handleSendMessage);
newChatButton.addEventListener('click', createNewConversation);
clearHistoryButton.addEventListener('click', clearAllHistory);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSendMessage();
    }
});

// Load conversations on startup
loadConversations();

// Add initial greeting only if there are no existing conversations
if (conversations.length === 0) {
    createNewConversation();
    addMessage('Hello! I am your Dashscope AI assistant. How can I help you today?', false);
}
