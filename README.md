# Prompt Optimizer Chrome Extension

A Chrome extension that automatically optimizes your prompts while typing in ChatGPT. Press `Alt+O` to instantly optimize your current prompt.

## Features

- **Seamless Integration**: Works directly inside ChatGPT where you're typing
- **Hotkey Activation**: Press `Alt+O` to optimize your current prompt
- **Visual Feedback**: Shows notifications for optimization status
- **Easy to Extend**: Simple to replace the stub optimization with real LLM API calls

## Installation

### Method 1: Load as Unpacked Extension (Recommended for Development)

1. **Download/Clone this repository** to your local machine
2. **Open Chrome** and navigate to `chrome://extensions/`
3. **Enable Developer Mode** by toggling the switch in the top-right corner
4. **Click "Load unpacked"** and select the folder containing the extension files
5. **Navigate to ChatGPT** (`https://chat.openai.com`) and start using the extension

### Method 2: Package and Install

1. **Package the extension**:
   - Go to `chrome://extensions/`
   - Click "Pack extension"
   - Select the extension folder
   - Click "Pack Extension"
   - This creates a `.crx` file

2. **Install the packaged extension**:
   - Drag the `.crx` file to `chrome://extensions/`
   - Click "Add extension" when prompted

## Usage

1. **Open ChatGPT** in your browser
2. **Type your prompt** in the message box
3. **Press `Alt+O`** to optimize your prompt
4. **Your prompt will be automatically replaced** with the optimized version

## File Structure

```
promptOptimizer/
├── manifest.json          # Extension manifest (Manifest V3)
├── background.js          # Background service worker
├── content.js            # Content script (runs in ChatGPT)
├── prompt_optimizer.js   # Optimization logic (standalone module)
├── popup.html            # Extension popup UI
├── popup.js              # Popup script
└── README.md             # This file
```

## Customization

### Adding Real LLM API Integration

To replace the stub optimization with a real LLM API:

1. **Open `content.js`**
2. **Find the `optimizePrompt` function** (around line 149)
3. **Replace the stub logic** with your API call:

```javascript
async function optimizePrompt(prompt) {
  const API_KEY = 'your-api-key-here';
  const API_URL = 'https://api.openai.com/v1/chat/completions';
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are a prompt optimization expert. Rewrite the given prompt to be more clear, specific, and effective while maintaining the original intent.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7
      })
    });
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    
    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
}
```

### Changing the Hotkey

To change the hotkey from `Alt+O` to something else:

1. **Open `content.js`**
2. **Find the `handleKeyDown` function** (around line 25)
3. **Modify the key combination**:

```javascript
function handleKeyDown(event) {
  // Change this line to your preferred key combination
  if (event.altKey && event.key === 'o') {  // Current: Alt+O
    event.preventDefault();
    optimizeCurrentPrompt();
  }
}
```

## Troubleshooting

### Extension Not Working

1. **Check if the extension is enabled** in `chrome://extensions/`
2. **Refresh ChatGPT** after installing the extension
3. **Check the browser console** for any error messages
4. **Ensure you're on the correct ChatGPT URL** (`https://chat.openai.com`)

### Hotkey Not Responding

1. **Make sure you're focused on the ChatGPT page**
2. **Try clicking in the text area first** before pressing the hotkey
3. **Check if another extension is using the same hotkey**

### Optimization Not Working

1. **Check the browser console** for error messages
2. **Verify the textarea is found** (the extension will show a notification)
3. **Make sure you have text in the prompt box** before pressing the hotkey

## Development

### Making Changes

1. **Edit the files** as needed
2. **Go to `chrome://extensions/`**
3. **Click the refresh icon** on the extension card
4. **Test your changes** in ChatGPT

### Debugging

1. **Open Chrome DevTools** (`F12`)
2. **Go to the Console tab**
3. **Look for messages** prefixed with "Prompt Optimizer"
4. **Check the Extensions tab** in DevTools for background script logs

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this extension!
