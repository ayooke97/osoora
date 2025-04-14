const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = 3000;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Proxy endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        
        // Set headers for SSE
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        // Format the prompt to request Markdown responses
        const formattedPrompt = `Please format your response using Markdown with proper headings, lists, code blocks, and other formatting where appropriate. Here's the user's message:\n\n${message}`;

        const response = await axios({
            method: 'post',
            url: 'https://dashscope-intl.aliyuncs.com/api/v1/apps/172f5d2e8d1b4b9da47dada83dcb7f19/completion',
            data: {
                input: {
                    prompt: formattedPrompt
                },
                parameters: {},
                debug: {}
            },
            headers: {
                'Authorization': `Bearer sk-0054022384a64f03abfdbcae8c001cbb`,
                'Content-Type': 'application/json',
                'X-DashScope-SSE': 'enable',
                'Accept': 'text/event-stream'
            },
            responseType: 'stream'
        });

        let buffer = '';
        let lastProcessedText = '';

        // Pipe the response stream to the client
        response.data.on('data', (chunk) => {
            try {
                buffer += chunk.toString();
                
                // Process complete events
                let newlineIndex;
                while ((newlineIndex = buffer.indexOf('\n\n')) !== -1) {
                    const event = buffer.slice(0, newlineIndex);
                    buffer = buffer.slice(newlineIndex + 2);

                    // Parse the event
                    const lines = event.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data:')) {
                            const jsonStr = line.slice(5).trim();
                            if (jsonStr && jsonStr !== '[DONE]') {
                                try {
                                    const jsonData = JSON.parse(jsonStr);
                                    if (jsonData.output?.text) {
                                        const currentText = jsonData.output.text;
                                        // Only send the new words
                                        if (currentText.startsWith(lastProcessedText) && currentText !== lastProcessedText) {
                                            const newText = currentText.slice(lastProcessedText.length);
                                            const formattedData = {
                                                choices: [{
                                                    message: {
                                                        content: newText
                                                    }
                                                }]
                                            };
                                            res.write(`data: ${JSON.stringify(formattedData)}\n\n`);
                                            lastProcessedText = currentText;
                                        } else if (currentText !== lastProcessedText) {
                                            // If there's a mismatch, send the full text
                                            const formattedData = {
                                                choices: [{
                                                    message: {
                                                        content: currentText
                                                    }
                                                }]
                                            };
                                            res.write(`data: ${JSON.stringify(formattedData)}\n\n`);
                                            lastProcessedText = currentText;
                                        }
                                    }
                                } catch (e) {
                                    console.error('Error parsing JSON:', e);
                                }
                            }
                        }
                    }
                }
            } catch (e) {
                console.error('Error processing chunk:', e);
            }
        });

        response.data.on('end', () => {
            res.write('data: [DONE]\n\n');
            res.end();
        });

        response.data.on('error', (error) => {
            console.error('Stream Error:', error);
            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
            res.end();
        });

    } catch (error) {
        console.error('Proxy Error:', error.response?.data || error.message);
        res.status(500).json({
            error: 'Error communicating with Dashscope API',
            details: error.response?.data || error.message
        });
    }
});

app.listen(port, () => {
    console.log(`Proxy server running at http://localhost:${port}`);
});
