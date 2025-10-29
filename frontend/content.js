// Content script for Prompt Optimizer Chrome Extension
// This runs inside ChatGPT and handles the actual prompt optimization

console.log('🚀 PROMPT OPTIMIZER: Content script loaded!');
console.log('🚀 PROMPT OPTIMIZER: Current URL:', window.location.href);
console.log('🚀 PROMPT OPTIMIZER: Document ready state:', document.readyState);

// Simple test - add a visible indicator
function addTestIndicator() {
  // Remove any existing indicator
  const existing = document.querySelector('#prompt-optimizer-test');
  if (existing) existing.remove();

  const testDiv = document.createElement('div');
  testDiv.id = 'prompt-optimizer-test';
  testDiv.innerHTML = '🚀 Prompt Optimizer Loaded!';
  testDiv.style.cssText = 'position: fixed; top: 10px; left: 10px; background: red; color: white; padding: 10px; z-index: 999999; font-size: 14px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);';
  document.body.appendChild(testDiv);
  console.log('🚀 PROMPT OPTIMIZER: Test indicator added!');

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (testDiv.parentNode) {
      testDiv.style.opacity = '0';
      testDiv.style.transition = 'opacity 0.5s';
      setTimeout(() => testDiv.remove(), 500);
    }
  }, 5000);
}

// Try multiple ways to add the indicator
function tryAddIndicator() {
  if (document.body) {
    addTestIndicator();
    return true;
  }
  return false;
}

// Try immediately
if (!tryAddIndicator()) {
  // Wait for DOM ready
  document.addEventListener('DOMContentLoaded', tryAddIndicator);

  // Also try after a delay
  setTimeout(() => {
    if (!document.querySelector('#prompt-optimizer-test')) {
      tryAddIndicator();
    }
  }, 1000);
}

// Wait for the page to be fully loaded
if (document.readyState === 'loading') {
  console.log('🚀 PROMPT OPTIMIZER: Document still loading, waiting for DOMContentLoaded');
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  console.log('🚀 PROMPT OPTIMIZER: Document already loaded, initializing immediately');
  initialize();
}

// Also try after a short delay in case the page is still loading
setTimeout(() => {
  console.log('🚀 PROMPT OPTIMIZER: Timeout check - trying to initialize again');
  if (!window.promptOptimizerInitialized) {
    initialize();
  }
}, 1000);

function initialize() {
  if (window.promptOptimizerInitialized) {
    console.log('Already initialized, skipping');
    return;
  }

  console.log('Prompt Optimizer content script loaded');
  console.log('Current URL:', window.location.href);

  // Mark as initialized
  window.promptOptimizerInitialized = true;

  // Listen for keyboard shortcuts
  document.addEventListener('keydown', handleKeyDown);
  console.log('Keyboard listener added');

  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Message received:', request);
    if (request.action === 'optimizePrompt') {
      optimizeCurrentPrompt();
      sendResponse({ success: true });
    }
  });
}

function handleKeyDown(event) {
  console.log('Key pressed:', event.key, 'Alt key:', event.altKey, 'Option key:', event.altKey);

  // Check for Alt+O (Windows/Linux) or Option+O (Mac) combination
  // On Mac, Option+O produces 'ø' or 'Ø', not 'o'
  if (event.altKey && (event.key === 'o' || event.key === 'O' || event.key === 'ø' || event.key === 'Ø')) {
    console.log('Option+O detected! Optimizing prompt...');
    event.preventDefault();
    optimizeCurrentPrompt();
  }

  // Check for Alt+J (Windows/Linux) or Option+J (Mac) combination
  // On Mac, Option+J produces '∆' or 'J'
  if (event.altKey && (event.key === 'j' || event.key === 'J' || event.key === '∆')) {
    console.log('Option+J detected! Creating JSON prompt...');
    event.preventDefault();
    createJSONPrompt();
  }
}

function optimizeCurrentPrompt() {
  const textarea = findPromptTextarea();

  if (!textarea) {
    console.log('No prompt textarea found');
    showNotification('No prompt found to optimize', 'error');
    return;
  }

  // Get text from textarea or contenteditable div
  let originalPrompt = '';

  try {
    if (textarea.tagName === 'TEXTAREA') {
      originalPrompt = textarea.value || '';
    } else {
      // For contenteditable divs, get text content
      originalPrompt = textarea.textContent || textarea.innerText || '';
    }
  } catch (error) {
    console.error('Error getting text from element:', error);
    originalPrompt = '';
  }

  // Ensure originalPrompt is a string
  originalPrompt = String(originalPrompt || '');

  if (!originalPrompt.trim()) {
    console.log('Empty prompt');
    showNotification('No text to optimize', 'warning');
    return;
  }

  console.log('Optimizing prompt:', originalPrompt);

  // Show loading state
  showNotification('Optimizing prompt...', 'info');

  // Optimize the prompt
  optimizePrompt(originalPrompt)
    .then(optimizedPrompt => {
      // Replace the textarea content
      if (textarea.tagName === 'TEXTAREA') {
        textarea.value = optimizedPrompt;
        // Trigger input event to update ChatGPT's state
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        textarea.dispatchEvent(new Event('change', { bubbles: true }));
      } else {
        // For contenteditable divs, replace the text content
        textarea.textContent = optimizedPrompt;
        textarea.innerHTML = optimizedPrompt;
        // Trigger input event
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        textarea.dispatchEvent(new Event('change', { bubbles: true }));
      }

      // Focus back to textarea
      textarea.focus();

      showNotification('Prompt optimized successfully!', 'success');
      console.log('Optimized prompt:', optimizedPrompt);
    })
    .catch(error => {
      console.error('Error optimizing prompt:', error);
      showNotification('Error optimizing prompt', 'error');
    });
}

function findPromptTextarea() {
  console.log('🔍 PROMPT OPTIMIZER: Looking for textarea...');

  // Try multiple selectors to find the ChatGPT prompt textarea
  const selectors = [
    'textarea[placeholder*="Message"]',
    'textarea[placeholder*="message"]',
    'textarea[data-id="root"]',
    'textarea[aria-label*="Message"]',
    'textarea[aria-label*="message"]',
    'textarea[placeholder*="Send a message"]',
    'textarea[placeholder*="send a message"]',
    'div[contenteditable="true"]',
    'textarea'
  ];

  for (const selector of selectors) {
    const elements = document.querySelectorAll(selector);
    console.log(`🔍 PROMPT OPTIMIZER: Found ${elements.length} elements for selector: ${selector}`);

    for (const element of elements) {
      if (element && element.offsetParent !== null) { // Check if visible
        console.log('🔍 PROMPT OPTIMIZER: Found visible textarea:', element);
        console.log('🔍 PROMPT OPTIMIZER: Textarea value:', element.value);
        return element;
      }
    }
  }

  console.log('🔍 PROMPT OPTIMIZER: No textarea found');
  return null;
}

function showNotification(message, type = 'info') {
  // Remove existing notifications
  const existing = document.querySelector('.prompt-optimizer-notification');
  if (existing) {
    existing.remove();
  }

  // Create notification element
  const notification = document.createElement('div');
  notification.className = 'prompt-optimizer-notification';
  notification.textContent = message;

  // Style the notification
  Object.assign(notification.style, {
    position: 'fixed',
    top: '20px',
    right: '20px',
    padding: '12px 16px',
    borderRadius: '8px',
    color: 'white',
    fontSize: '14px',
    fontWeight: '500',
    zIndex: '10000',
    maxWidth: '300px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    transition: 'all 0.3s ease',
    backgroundColor: type === 'error' ? '#ef4444' :
                    type === 'warning' ? '#f59e0b' :
                    type === 'success' ? '#10b981' : '#3b82f6'
  });

  document.body.appendChild(notification);

  // Auto-remove after 3 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.style.opacity = '0';
      notification.style.transform = 'translateY(-10px)';
      setTimeout(() => notification.remove(), 300);
    }
  }, 3000);
}

/**
 * Optimizes a given prompt using a free AI model
 * @param {string} prompt - The original prompt to optimize
 * @returns {Promise<string>} - The optimized prompt
 */
async function optimizePrompt(prompt) {
  // System prompt for optimization
  const SYSTEM_PROMPT = `You are a prompt optimization expert. Your ONLY job is to rewrite user prompts to make them more effective for AI language models.

CRITICAL INSTRUCTIONS:
- You are NOT answering the user's prompt
- You are NOT providing solutions to their questions
- You are ONLY rewriting their prompt to be better
- Return ONLY the improved version of their prompt
- Do NOT add explanations, commentary, or meta-text

When optimizing, make the prompt:
1. More specific and detailed
2. Better structured and organized
3. Clearer about what response is needed
4. More likely to get high-quality AI responses

Example:
User prompt: "help me write code"
Optimized: "Please help me write clean, well-documented code. I need assistance with [specific programming language/task]. Please include comments explaining the logic and best practices. The code should be production-ready and follow industry standards."

Remember: You are optimizing the prompt, not answering it!`;

  try {
    // Try Groq first (free tier, very fast)
    console.log('🚀 PROMPT OPTIMIZER: Calling Groq API...');
    const result = await callGroqAPI(prompt, SYSTEM_PROMPT);
    console.log('🚀 PROMPT OPTIMIZER: Groq API success:', result);
    return result;
  } catch (error) {
    console.error('🚀 PROMPT OPTIMIZER: Groq API failed, trying fallback:', error);

    // Fallback to local optimization if API fails
    console.log('🚀 PROMPT OPTIMIZER: Using local optimization fallback');
    return await localOptimization(prompt);
  }
}

/**
 * Call Groq API for optimization
 */
async function callGroqAPI(prompt, systemPrompt) {
  const API_KEY = 'YOUR_GROQ_API_KEY_HERE'; // Replace with your Groq API key
  const API_URL = 'https://api.groq.com/openai/v1/chat/completions';

  console.log('🚀 PROMPT OPTIMIZER: Making API request to Groq...');
  console.log('🚀 PROMPT OPTIMIZER: API Key length:', API_KEY.length);

  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama-3.1-8b-instant',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Please optimize this prompt: "${prompt}"` }
      ],
      max_tokens: 800,
      temperature: 0.3,
      top_p: 0.9,
      stop: ["User prompt:", "Optimized:", "Remember:"]
    })
  });

  console.log('🚀 PROMPT OPTIMIZER: API response status:', response.status);

  if (!response.ok) {
    const errorText = await response.text();
    console.error('🚀 PROMPT OPTIMIZER: API error response:', errorText);
    throw new Error(`Groq API request failed: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  console.log('🚀 PROMPT OPTIMIZER: API response data:', data);

  let optimizedPrompt = data.choices[0].message.content.trim();

  // Clean up the response to ensure it's just the optimized prompt
  // Remove any meta-commentary or explanations
  const lines = optimizedPrompt.split('\n');
  const cleanedLines = lines.filter(line => {
    const trimmed = line.trim();
    return !trimmed.startsWith('User prompt:') &&
           !trimmed.startsWith('Optimized:') &&
           !trimmed.startsWith('Remember:') &&
           !trimmed.startsWith('Here') &&
           !trimmed.startsWith('The optimized') &&
           trimmed.length > 0;
  });

  optimizedPrompt = cleanedLines.join('\n').trim();

  // If the response seems to be answering the prompt instead of optimizing it,
  // fall back to local optimization
  if (optimizedPrompt.length < 10 || optimizedPrompt.includes('I can help') || optimizedPrompt.includes('I\'ll help')) {
    console.log('🚀 PROMPT OPTIMIZER: Response seems to be answering instead of optimizing, using fallback');
    return await localOptimization(prompt);
  }

  return optimizedPrompt;
}

/**
 * Fallback local optimization (no API needed)
 */
async function localOptimization(prompt) {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 500));

  // Simple local optimization rules
  let optimized = prompt.trim();

  // Add structure if missing
  if (!optimized.includes('Please') && !optimized.includes('Can you') && !optimized.includes('I need')) {
    optimized = `Please ${optimized.toLowerCase()}`;
  }

  // Add specificity if too vague
  if (optimized.length < 50) {
    optimized += `. Please provide specific examples and detailed explanations.`;
  }

  // Add formatting if it's a list
  if (optimized.includes(' and ') && !optimized.includes('\n')) {
    optimized = optimized.replace(/ and /g, '\n- ');
    optimized = 'Please provide:\n' + optimized;
  }

  return optimized + ' [Optimized]';
}

/**
 * Creates a JSON prompt from the current text
 */
function createJSONPrompt() {
  const textarea = findPromptTextarea();

  if (!textarea) {
    console.log('No prompt textarea found');
    showNotification('No prompt found to convert', 'error');
    return;
  }

  // Get text from textarea or contenteditable div
  let originalPrompt = '';

  try {
    if (textarea.tagName === 'TEXTAREA') {
      originalPrompt = textarea.value || '';
    } else {
      // For contenteditable divs, get text content
      originalPrompt = textarea.textContent || textarea.innerText || '';
    }
  } catch (error) {
    console.error('Error getting text from element:', error);
    originalPrompt = '';
  }

  // Ensure originalPrompt is a string
  originalPrompt = String(originalPrompt || '');

  if (!originalPrompt.trim()) {
    console.log('Empty prompt');
    showNotification('No text to convert to JSON', 'warning');
    return;
  }

  console.log('Creating JSON prompt for:', originalPrompt);

  // Show loading state
  showNotification('Creating JSON prompt...', 'info');

  // Create JSON prompt
  const jsonPrompt = createJSONFromPrompt(originalPrompt);

  // Replace the textarea content
  if (textarea.tagName === 'TEXTAREA') {
    textarea.value = jsonPrompt;
    // Trigger input event to update ChatGPT's state
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    // For contenteditable divs, replace the text content
    textarea.textContent = jsonPrompt;
    textarea.innerHTML = jsonPrompt;
    // Trigger input event
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.dispatchEvent(new Event('change', { bubbles: true }));
  }

  // Focus back to textarea
  textarea.focus();

  showNotification('JSON prompt created successfully!', 'success');
  console.log('JSON prompt created:', jsonPrompt);
}

/**
 * Converts a regular prompt into a structured JSON prompt
 */
function createJSONFromPrompt(prompt) {
  // Clean the prompt
  const cleanPrompt = prompt.trim();

  // Create a structured JSON prompt
  const jsonPrompt = {
    "role": "user",
    "content": cleanPrompt,
    "context": {
      "task_type": "general_assistance",
      "expected_output": "detailed_response",
      "format_preferences": {
        "structure": "clear_and_organized",
        "detail_level": "comprehensive",
        "examples": "when_helpful"
      }
    },
    "instructions": {
      "response_style": "professional_and_helpful",
      "include_examples": true,
      "be_specific": true,
      "provide_context": true
    }
  };

  // Convert to formatted JSON string
  return JSON.stringify(jsonPrompt, null, 2);
}

/**
 * Example function showing how to integrate with a real LLM API
 * Uncomment and modify this when you're ready to use a real API
 */
/*
async function callOptimizationAPI(prompt) {
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
*/
