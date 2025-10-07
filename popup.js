// Popup script for Prompt Optimizer Chrome Extension
// This handles the popup UI and status updates

document.addEventListener('DOMContentLoaded', () => {
  updateStatus();
  
  // Check if we're on ChatGPT
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentTab = tabs[0];
    if (currentTab && currentTab.url && currentTab.url.includes('chat.openai.com')) {
      setStatus('Ready to optimize prompts', 'active');
    } else {
      setStatus('Please navigate to ChatGPT to use this extension', 'inactive');
    }
  });
});

function updateStatus() {
  // This could be expanded to check extension status
  // For now, we'll just show a basic status
}

function setStatus(message, type) {
  const statusElement = document.getElementById('status');
  const statusText = document.getElementById('statusText');
  
  statusElement.className = `status ${type}`;
  statusText.textContent = message;
}
