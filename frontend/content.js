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

  initializeDOMWatcher();
  initializePromptTracking();
  loadFailedRequests();
  
  if (domWatcherState.retryIntervalId) {
    clearInterval(domWatcherState.retryIntervalId);
  }
  domWatcherState.retryIntervalId = setInterval(retryFailedRequests, 60000);
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
      if (textarea.dataset) {
        textarea.dataset.optimizedPrompt = optimizedPrompt;
      }

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

const domWatcherState = {
  observer: null,
  threadContainer: null,
  currentConversationId: null,
  capturedMessageIds: new Set(),
  isWatching: false,
  streamingResponses: new Map(),
  failedRequests: [],
  retryAttempts: new Map(),
  retryIntervalId: null,
  urlCheckIntervalId: null
};

function getConversationId() {
  const pathname = window.location.pathname;
  const match = pathname.match(/^\/c\/([a-f0-9-]+)$/);
  return match ? match[1] : null;
}

function generatePromptId() {
  return `prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getSessionId() {
  if (!window.promptOptimizerSessionId) {
    window.promptOptimizerSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  return window.promptOptimizerSessionId;
}

function initializeDOMWatcher() {
  console.log('🔍 DOM WATCHER: Initializing...');

  const conversationId = getConversationId();
  if (!conversationId) {
    console.log('🔍 DOM WATCHER: Not on a ChatGPT conversation page, skipping');
    return;
  }

  domWatcherState.currentConversationId = conversationId;
  console.log('🔍 DOM WATCHER: Current conversation ID:', conversationId);

  const waitForThread = setInterval(() => {
    const threadContainer = document.querySelector('#thread');
    if (threadContainer) {
      clearInterval(waitForThread);
      domWatcherState.threadContainer = threadContainer;
      startWatchingThread();
    }
  }, 500);

  setTimeout(() => {
    clearInterval(waitForThread);
    if (!domWatcherState.threadContainer) {
      console.warn('🔍 DOM WATCHER: Thread container not found after 10 seconds');
    }
  }, 10000);

  watchForConversationChanges();
}

function startWatchingThread() {
  if (domWatcherState.isWatching) {
    console.log('🔍 DOM WATCHER: Already watching, skipping');
    return;
  }

  const threadContainer = domWatcherState.threadContainer;
  if (!threadContainer) {
    console.error('🔍 DOM WATCHER: Thread container not found');
    setTimeout(() => {
      const retryContainer = document.querySelector('#thread');
      if (retryContainer) {
        domWatcherState.threadContainer = retryContainer;
        startWatchingThread();
      }
    }, 2000);
    return;
  }

  console.log('🔍 DOM WATCHER: Starting to watch thread container');

  try {
    domWatcherState.observer = new MutationObserver((mutations) => {
      try {
        handleThreadMutations(mutations);
      } catch (error) {
        console.error('🔍 DOM WATCHER: Error in mutation handler:', error);
      }
    });

    domWatcherState.observer.observe(threadContainer, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: false
    });

    domWatcherState.isWatching = true;
    console.log('🔍 DOM WATCHER: Observer started successfully');
  } catch (error) {
    console.error('🔍 DOM WATCHER: Error starting observer:', error);
    domWatcherState.isWatching = false;
  }
}

function handleThreadMutations(mutations) {
  for (const mutation of mutations) {
    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          checkForNewResponse(node);
        }
      }
    }

    if (mutation.type === 'characterData' || mutation.type === 'childList') {
      const allNodes = [
        ...(mutation.addedNodes || []),
        mutation.target
      ];

      for (const node of allNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const responseElement = node.closest('[data-message-author-role="assistant"]');
          if (responseElement) {
            handleStreamingResponse(responseElement);
          }
        }
      }
    }
  }
}

function checkForNewResponse(element) {
  const responseElement = element.querySelector?.('[data-message-author-role="assistant"]') ||
                         (element.matches?.('[data-message-author-role="assistant"]') ? element : null);

  if (responseElement) {
    const messageId = responseElement.getAttribute('data-message-id');
    
    if (messageId && domWatcherState.capturedMessageIds.has(messageId)) {
      return;
    }

    console.log('🔍 DOM WATCHER: New ChatGPT response detected:', messageId);
    
    if (messageId) {
      domWatcherState.capturedMessageIds.add(messageId);
    }

    setTimeout(() => {
      handleStreamingResponse(responseElement);
    }, 1000);
  }
}

function handleStreamingResponse(responseElement) {
  const messageId = responseElement.getAttribute('data-message-id');
  
  if (!messageId) {
    return;
  }

  if (!domWatcherState.streamingResponses.has(messageId)) {
    console.log('🔍 DOM WATCHER: Starting to track streaming response:', messageId);
    domWatcherState.streamingResponses.set(messageId, {
      element: responseElement,
      lastText: '',
      lastChangeTime: Date.now(),
      completionTimer: null
    });
  }

  const streamingData = domWatcherState.streamingResponses.get(messageId);
  const markdownElement = responseElement.querySelector('.markdown');
  
  if (!markdownElement) {
    return;
  }

  const currentText = markdownElement.innerText || markdownElement.textContent || '';

  if (currentText !== streamingData.lastText) {
    streamingData.lastText = currentText;
    streamingData.lastChangeTime = Date.now();

    if (streamingData.completionTimer) {
      clearTimeout(streamingData.completionTimer);
    }

    streamingData.completionTimer = setTimeout(() => {
      checkResponseComplete(messageId);
    }, 2500);
  }
}

function checkResponseComplete(messageId) {
  const streamingData = domWatcherState.streamingResponses.get(messageId);
  
  if (!streamingData) {
    return;
  }

  const timeSinceLastChange = Date.now() - streamingData.lastChangeTime;
  
  if (timeSinceLastChange >= 2000) {
    console.log('🔍 DOM WATCHER: Response complete:', messageId);
    extractAndProcessResponse(streamingData.element, messageId);
    domWatcherState.streamingResponses.delete(messageId);
  }
}

function extractAndProcessResponse(responseElement, messageId) {
  try {
    const markdownElement = responseElement.querySelector('.markdown');
    
    if (!markdownElement) {
      console.warn('🔍 DOM WATCHER: Markdown element not found, trying alternative selectors');
      const altSelectors = [
        '[class*="markdown"]',
        '[class*="prose"]',
        'div > p',
        'p'
      ];
      
      for (const selector of altSelectors) {
        const element = responseElement.querySelector(selector);
        if (element && (element.innerText || element.textContent)) {
          return extractFromElement(element, messageId);
        }
      }
      
      console.warn('🔍 DOM WATCHER: Could not find text element for message:', messageId);
      return;
    }

    extractFromElement(markdownElement, messageId);
  } catch (error) {
    console.error('🔍 DOM WATCHER: Error extracting response:', error);
  }
}

function extractFromElement(element, messageId) {
  const responseText = element.innerText || element.textContent || '';
  
  if (!responseText.trim()) {
    console.warn('🔍 DOM WATCHER: Empty response text for message:', messageId);
    return;
  }

  const conversationId = getConversationId();
  if (!conversationId) {
    console.warn('🔍 DOM WATCHER: No conversation ID, cannot process response');
    return;
  }

  const responseData = {
    message_id: messageId,
    conversation_id: conversationId,
    chatgpt_output: responseText,
    timestamp: Date.now()
  };

  console.log('🔍 DOM WATCHER: Response extracted:', {
    message_id: messageId,
    conversation_id: conversationId,
    text_length: responseText.length
  });

  matchResponseToPrompt(responseData);
}

function matchResponseToPrompt(responseData) {
  const conversationId = responseData.conversation_id;
  const allPrompts = getPendingPrompts(conversationId);
  
  if (!allPrompts || allPrompts.length === 0) {
    console.warn('🔍 DOM WATCHER: No pending prompts found for conversation:', conversationId);
    return;
  }

  const responseTimestamp = responseData.timestamp;
  
  const validPrompts = allPrompts.filter(prompt => {
    const timeDiff = responseTimestamp - prompt.timestamp;
    return timeDiff >= 0 && timeDiff < 300000;
  });

  if (validPrompts.length === 0) {
    console.warn('🔍 DOM WATCHER: No valid prompts found (all are after response or too old)');
    return;
  }

  validPrompts.sort((a, b) => b.timestamp - a.timestamp);
  
  let matchedPrompt = null;
  
  for (const prompt of validPrompts) {
    const timeDiff = responseTimestamp - prompt.timestamp;
    
    if (timeDiff >= 0 && timeDiff < 300000) {
      matchedPrompt = prompt;
      break;
    }
  }

  if (!matchedPrompt) {
    console.warn('🔍 DOM WATCHER: Could not find valid prompt to match');
    return;
  }

  const responseTime = responseData.timestamp - matchedPrompt.timestamp;
  
  if (responseTime < 0) {
    console.warn('🔍 DOM WATCHER: Negative response time, skipping match');
    return;
  }

  if (responseTime > 300000) {
    console.warn('🔍 DOM WATCHER: Response time too long (>5 minutes), likely wrong match');
    return;
  }

  const matchedData = {
    prompt_id: matchedPrompt.prompt_id,
    conversation_id: conversationId,
    chatgpt_output: responseData.chatgpt_output,
    chatgpt_response_time: responseTime,
    message_id: responseData.message_id,
    session_id: matchedPrompt.session_id,
    original_prompt: matchedPrompt.original_prompt,
    optimized_prompt: matchedPrompt.optimized_prompt,
    timestamp: responseData.timestamp
  };

  console.log('🔍 DOM WATCHER: Response matched to prompt:', {
    prompt_id: matchedPrompt.prompt_id,
    response_time_ms: responseTime,
    original_prompt_length: matchedPrompt.original_prompt.length
  });

  removeMatchedPrompt(conversationId, matchedPrompt.prompt_id);
  
  sendResponseToBackend(matchedData);
}

function removeMatchedPrompt(conversationId, promptId) {
  const prompts = promptTrackingState.pendingPrompts.get(conversationId);
  if (!prompts) {
    return;
  }

  const index = prompts.findIndex(p => p.prompt_id === promptId);
  if (index !== -1) {
    prompts.splice(index, 1);
    console.log('🔍 DOM WATCHER: Removed matched prompt from pending list');
    
    if (prompts.length === 0) {
      promptTrackingState.pendingPrompts.delete(conversationId);
    }
  }
}

function sendResponseToBackend(data, retryCount = 0) {
  const backendUrl = 'http://localhost:8000/api/v1/save-chatgpt-response';
  const maxRetries = 3;
  const retryDelay = 2000;
  
  const requestData = {
    prompt_id: data.prompt_id,
    conversation_id: data.conversation_id,
    chatgpt_output: data.chatgpt_output,
    chatgpt_response_time: data.chatgpt_response_time,
    message_id: data.message_id,
    session_id: data.session_id,
    original_prompt: data.original_prompt,
    optimized_prompt: data.optimized_prompt,
  };

  fetch(backendUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(result => {
      console.log('🔍 DOM WATCHER: Response saved to backend:', result);
      const key = data.message_id || data.prompt_id;
      domWatcherState.retryAttempts.delete(key);
      removeFromFailedQueue(key);
    })
    .catch(error => {
      console.error('🔍 DOM WATCHER: Error sending to backend:', error);
      
      if (retryCount < maxRetries) {
        const key = data.message_id || data.prompt_id;
        const currentAttempts = domWatcherState.retryAttempts.get(key) || 0;
        domWatcherState.retryAttempts.set(key, currentAttempts + 1);
        
        console.log(`🔍 DOM WATCHER: Retrying in ${retryDelay}ms (attempt ${retryCount + 1}/${maxRetries})`);
        setTimeout(() => {
          sendResponseToBackend(data, retryCount + 1);
        }, retryDelay * (retryCount + 1));
      } else {
        console.warn('🔍 DOM WATCHER: Max retries reached, adding to failed queue');
        addToFailedQueue(data);
      }
    });
}

function addToFailedQueue(data) {
  const key = data.message_id || data.prompt_id;
  const existing = domWatcherState.failedRequests.find(r => 
    (r.message_id || r.prompt_id) === key
  );
  
  if (!existing) {
    const maxRetriesPerRequest = 10;
    const newRequest = {
      ...data,
      failed_at: Date.now(),
      retry_count: 0,
      max_retries: maxRetriesPerRequest
    };
    
    domWatcherState.failedRequests.push(newRequest);
    
    try {
      localStorage.setItem('promptOptimizer_failedRequests', 
        JSON.stringify(domWatcherState.failedRequests));
    } catch (e) {
      console.warn('🔍 DOM WATCHER: Could not save to localStorage:', e);
    }
  } else {
    existing.failed_at = Date.now();
  }
}

function removeFromFailedQueue(key) {
  domWatcherState.failedRequests = domWatcherState.failedRequests.filter(r => 
    (r.message_id || r.prompt_id) !== key
  );
  
  try {
    localStorage.setItem('promptOptimizer_failedRequests', 
      JSON.stringify(domWatcherState.failedRequests));
  } catch (e) {
    console.warn('🔍 DOM WATCHER: Could not update localStorage:', e);
  }
}

function retryFailedRequests() {
  if (domWatcherState.failedRequests.length === 0) {
    return;
  }

  const now = Date.now();
  const maxAge = 24 * 60 * 60 * 1000;
  const maxRetriesPerRequest = 10;
  const maxStorageSize = 50;
  
  if (domWatcherState.failedRequests.length > maxStorageSize) {
    domWatcherState.failedRequests = domWatcherState.failedRequests
      .sort((a, b) => (b.failed_at || 0) - (a.failed_at || 0))
      .slice(0, maxStorageSize);
    console.warn(`🔍 DOM WATCHER: Trimmed failed requests to ${maxStorageSize} most recent`);
  }
  
  domWatcherState.failedRequests = domWatcherState.failedRequests.filter(request => {
    const age = now - request.failed_at;
    const retryCount = request.retry_count || 0;
    
    if (age > maxAge) {
      console.log('🔍 DOM WATCHER: Removing request older than 24 hours');
      return false;
    }
    
    if (retryCount >= maxRetriesPerRequest) {
      console.log('🔍 DOM WATCHER: Removing request that exceeded max retries');
      return false;
    }
    
    request.retry_count = retryCount + 1;
    sendResponseToBackend(request, 0);
    return true;
  });
  
  try {
    const data = JSON.stringify(domWatcherState.failedRequests);
    if (data.length > 5 * 1024 * 1024) {
      console.warn('🔍 DOM WATCHER: localStorage data too large, clearing old entries');
      domWatcherState.failedRequests = domWatcherState.failedRequests.slice(-10);
    }
    localStorage.setItem('promptOptimizer_failedRequests', 
      JSON.stringify(domWatcherState.failedRequests));
  } catch (e) {
    if (e.name === 'QuotaExceededError') {
      console.warn('🔍 DOM WATCHER: localStorage full, clearing old entries');
      domWatcherState.failedRequests = domWatcherState.failedRequests.slice(-10);
      try {
        localStorage.setItem('promptOptimizer_failedRequests', 
          JSON.stringify(domWatcherState.failedRequests));
      } catch (e2) {
        console.error('🔍 DOM WATCHER: Could not save to localStorage:', e2);
        domWatcherState.failedRequests = [];
      }
    } else {
      console.warn('🔍 DOM WATCHER: Could not update localStorage:', e);
    }
  }
}

function loadFailedRequests() {
  try {
    const stored = localStorage.getItem('promptOptimizer_failedRequests');
    if (stored) {
      domWatcherState.failedRequests = JSON.parse(stored);
      if (domWatcherState.failedRequests.length > 0) {
        console.log(`🔍 DOM WATCHER: Loaded ${domWatcherState.failedRequests.length} failed requests`);
        setTimeout(retryFailedRequests, 5000);
      }
    }
  } catch (e) {
    console.warn('🔍 DOM WATCHER: Could not load failed requests:', e);
  }
}

function watchForConversationChanges() {
  if (domWatcherState.urlCheckIntervalId) {
    clearInterval(domWatcherState.urlCheckIntervalId);
  }

  let lastUrl = window.location.href;

  const checkUrl = () => {
    const currentUrl = window.location.href;
    if (currentUrl !== lastUrl) {
      lastUrl = currentUrl;
      handleConversationChange();
    }
  };

  domWatcherState.urlCheckIntervalId = setInterval(checkUrl, 1000);

  if (!window.promptOptimizerPopStateHandler) {
    window.promptOptimizerPopStateHandler = handleConversationChange;
    window.addEventListener('popstate', window.promptOptimizerPopStateHandler);
  }

  if (!window.promptOptimizerHistoryPatched) {
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function(...args) {
      originalPushState.apply(history, args);
      setTimeout(handleConversationChange, 100);
    };

    history.replaceState = function(...args) {
      originalReplaceState.apply(history, args);
      setTimeout(handleConversationChange, 100);
    };

    window.promptOptimizerHistoryPatched = true;
  }
}

function handleConversationChange() {
  const newConversationId = getConversationId();
  
  if (newConversationId !== domWatcherState.currentConversationId) {
    console.log('🔍 DOM WATCHER: Conversation changed:', {
      old: domWatcherState.currentConversationId,
      new: newConversationId
    });

    domWatcherState.currentConversationId = newConversationId;

    if (domWatcherState.threadContainer) {
      if (domWatcherState.observer) {
        domWatcherState.observer.disconnect();
        domWatcherState.isWatching = false;
      }

      domWatcherState.capturedMessageIds.clear();
      domWatcherState.streamingResponses.clear();
      domWatcherState.retryAttempts.clear();

      startWatchingThread();
    } else {
      initializeDOMWatcher();
    }
  }
}

const promptTrackingState = {
  pendingPrompts: new Map(),
  maxPendingAge: 30 * 60 * 1000
};

function initializePromptTracking() {
  console.log('📝 PROMPT TRACKER: Initializing...');

  const textarea = findPromptTextarea();
  if (!textarea) {
    console.log('📝 PROMPT TRACKER: Textarea not found, will retry');
    setTimeout(initializePromptTracking, 2000);
    return;
  }

  trackPromptSubmission(textarea);
  
  console.log('📝 PROMPT TRACKER: Initialized successfully');
}

function trackPromptSubmission(textarea) {
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      setTimeout(() => {
        capturePrompt(textarea);
      }, 100);
    }
  });

  const sendButton = findSendButton();
  if (sendButton) {
    sendButton.addEventListener('click', () => {
      setTimeout(() => {
        capturePrompt(textarea);
      }, 100);
    });
  }

  const form = textarea.closest('form');
  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      capturePrompt(textarea);
      setTimeout(() => {
        form.submit();
      }, 50);
    });
  }

  console.log('📝 PROMPT TRACKER: Event listeners added');
}

function findSendButton() {
  const selectors = [
    'button[type="submit"]',
    'button[aria-label*="Send"]',
    'button[aria-label*="send"]',
    'button:has(svg)',
    '[data-testid*="send"]',
    'button:last-of-type'
  ];

  for (const selector of selectors) {
    const buttons = document.querySelectorAll(selector);
    for (const button of buttons) {
      if (button.offsetParent !== null) {
        const text = button.textContent?.toLowerCase() || '';
        const ariaLabel = button.getAttribute('aria-label')?.toLowerCase() || '';
        
        if (text.includes('send') || ariaLabel.includes('send') || 
            button.querySelector('svg')) {
          console.log('📝 PROMPT TRACKER: Found send button:', button);
          return button;
        }
      }
    }
  }

  return null;
}

function capturePrompt(textarea) {
  let promptText = '';
  try {
    if (textarea.tagName === 'TEXTAREA') {
      promptText = textarea.value || '';
    } else {
      promptText = textarea.textContent || textarea.innerText || '';
    }
  } catch (error) {
    console.error('📝 PROMPT TRACKER: Error getting prompt text:', error);
    return;
  }

  promptText = String(promptText || '').trim();

  if (!promptText) {
    console.log('📝 PROMPT TRACKER: Empty prompt, skipping');
    return;
  }

  const conversationId = getConversationId();
  if (!conversationId) {
    console.warn('📝 PROMPT TRACKER: No conversation ID, skipping');
    return;
  }

  const promptId = generatePromptId();
  const timestamp = Date.now();
  const sessionId = getSessionId();

  let optimizedPrompt = promptText;
  if (textarea.dataset?.optimizedPrompt) {
    optimizedPrompt = textarea.dataset.optimizedPrompt;
  }

  const promptData = {
    prompt_id: promptId,
    conversation_id: conversationId,
    original_prompt: promptText,
    optimized_prompt: optimizedPrompt,
    timestamp: timestamp,
    session_id: sessionId
  };

  if (!promptTrackingState.pendingPrompts.has(conversationId)) {
    promptTrackingState.pendingPrompts.set(conversationId, []);
  }

  const prompts = promptTrackingState.pendingPrompts.get(conversationId);
  prompts.push(promptData);

  console.log('📝 PROMPT TRACKER: Prompt captured:', {
    prompt_id: promptId,
    conversation_id: conversationId,
    prompt_length: promptText.length,
    total_pending: prompts.length
  });

  cleanupOldPrompts();
}

function cleanupOldPrompts() {
  const now = Date.now();
  const maxAge = promptTrackingState.maxPendingAge;

  for (const [conversationId, prompts] of promptTrackingState.pendingPrompts.entries()) {
    const filtered = prompts.filter(prompt => {
      const age = now - prompt.timestamp;
      return age < maxAge;
    });

    if (filtered.length === 0) {
      promptTrackingState.pendingPrompts.delete(conversationId);
    } else {
      promptTrackingState.pendingPrompts.set(conversationId, filtered);
    }
  }
}

function getMostRecentPrompt(conversationId) {
  const prompts = promptTrackingState.pendingPrompts.get(conversationId);
  if (!prompts || prompts.length === 0) {
    return null;
  }

  return prompts[prompts.length - 1];
}

function getPendingPrompts(conversationId) {
  return promptTrackingState.pendingPrompts.get(conversationId) || [];
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
