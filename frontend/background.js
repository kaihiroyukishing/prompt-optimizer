// Background script for Prompt Optimizer Chrome Extension
// This handles keyboard shortcuts and communication between content script and popup

chrome.runtime.onInstalled.addListener(() => {
  console.log('Prompt Optimizer extension installed');
});

// Handle keyboard shortcuts
chrome.commands?.onCommand.addListener((command) => {
  if (command === 'optimize-prompt') {
    // Send message to content script to optimize the current prompt
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.url?.includes('chat.openai.com')) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'optimizePrompt' });
      }
    });
  }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'optimizePrompt') {
    // This could be expanded to call an external API
    // For now, we'll let the content script handle the optimization
    sendResponse({ success: true });
  }
});
