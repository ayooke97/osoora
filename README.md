# Dashscope AI Chatbot

A modern, responsive chatbot application powered by Dashscope's AI API with markdown formatting support.

## Features

- 🤖 Real-time AI chat interface
- 📝 Markdown formatting for bot responses
- 💻 Syntax highlighting for code blocks
- 🎨 Modern and responsive UI
- 🔒 Secure API handling through proxy server
- 📱 Mobile-friendly design

## Tech Stack

- Frontend:
  - HTML5
  - CSS3
  - JavaScript (ES6+)
  - Axios for API requests
  - Marked.js for markdown parsing
  - Highlight.js for code syntax highlighting

- Backend:
  - Node.js
  - Express.js
  - CORS middleware
  - Dashscope AI API integration

## Prerequisites

- Node.js (v14 or higher)
- npm (Node Package Manager)
- Dashscope API credentials
  - API Key
  - App ID

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd osoora
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure the application:
   - The API key and App ID are pre-configured in `server.js`
   - Modify if needed for your own Dashscope credentials

4. Start the server:
   ```bash
   node server.js
   ```

5. Access the application:
   - Open your browser
   - Navigate to `http://localhost:3000`

## Project Structure

```
osoora/
├── index.html      # Main HTML file
├── style.css       # Styles and theme
├── script.js       # Frontend logic
├── server.js       # Proxy server
├── package.json    # Dependencies
├── LICENSE         # Proprietary license
└── README.md       # Documentation
```

## Usage

1. Start the server using `node server.js`
2. Open the application in your browser
3. Type your message in the input field
4. Press Enter or click Send
5. View the AI's response with markdown formatting

## Features in Detail

### Markdown Support
The chatbot supports full markdown formatting including:
- Headers (h1-h6)
- Lists (ordered and unordered)
- Code blocks with syntax highlighting
- Blockquotes
- Links (opening in new tabs)
- Tables
- Inline code
- And more!

### Security
- API keys are secured on the server side
- CORS protection
- XSS prevention
- Secure external links

## License

This project is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited. All rights reserved.

## Author

[Your Name/Organization]

## Support

For support or inquiries, please contact [your contact information].
