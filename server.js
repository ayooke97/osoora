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
        
        const response = await axios.post(
            'https://dashscope-intl.aliyuncs.com/api/v1/apps/172f5d2e8d1b4b9da47dada83dcb7f19/completion',
            {
                input: {
                    prompt: message
                },
                parameters: {},
                debug: {}
            },
            {
                headers: {
                    'Authorization': `Bearer sk-0054022384a64f03abfdbcae8c001cbb`,
                    'Content-Type': 'application/json',
                }
            }
        );
        
        res.json(response.data);
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
