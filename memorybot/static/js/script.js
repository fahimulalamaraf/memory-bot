document.addEventListener("DOMContentLoaded", () => {
  /* === Chat Functionality === */
  const chatContainer = document.getElementById("chat-container");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");

  if (sendBtn) {
    sendBtn.addEventListener("click", () => {
      const message = chatInput.value.trim();
      if (message) {
        // Append user message
        const userMsg = document.createElement("div");
        userMsg.classList.add("chat-message", "user-message");
        userMsg.textContent = message;
        chatContainer.appendChild(userMsg);
        chatInput.value = "";
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Simulate bot response
        setTimeout(() => {
          const botMsg = document.createElement("div");
          botMsg.classList.add("chat-message", "bot-message");
          botMsg.textContent = "Bot: This is a simulated response!";
          chatContainer.appendChild(botMsg);
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 1000);
      }
    });
  }

  /* === Upload Functionality === */
  const uploadContainer = document.getElementById("upload-container");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("upload-preview");

  if (uploadContainer) {
    uploadContainer.addEventListener("click", () => fileInput.click());

    uploadContainer.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadContainer.classList.add("dragover");
    });

    uploadContainer.addEventListener("dragleave", () => {
      uploadContainer.classList.remove("dragover");
    });

    uploadContainer.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadContainer.classList.remove("dragover");
      displayFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener("change", (e) => {
      displayFiles(e.target.files);
    });
  }

  function displayFiles(files) {
    previewContainer.innerHTML = "";
    Array.from(files).forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = document.createElement("img");
        img.src = e.target.result;
        previewContainer.appendChild(img);
      };
      reader.readAsDataURL(file);
    });
  }
});

async function sendMessage(event) {
    event.preventDefault();
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    const typingIndicator = document.getElementById('typing-indicator');
    
    if (!message && !currentImageData) {
        return;
    }
    
    // Show user message immediately
    appendMessage({
        text: message,
        image: currentImageData
    }, true);
    
    // Clear input and show typing indicator BEFORE the API call
    messageInput.value = '';
    typingIndicator.classList.remove('hidden');
    
    // Ensure the typing indicator is visible by scrolling to bottom
    const chatMessages = document.getElementById('chat-messages');
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
    
    try {
        const formData = new FormData();
        formData.append('message', message);
        
        if (currentImageData) {
            const response = await fetch(currentImageData);
            const blob = await response.blob();
            formData.append('image', blob, 'image.jpg');
        }
        
        // Add a small delay to ensure typing indicator is visible
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const response = await fetch('/api/v1/chat/message', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        typingIndicator.classList.add('hidden');
        
        // Show bot response
        appendMessage(data);
        
        if (currentImageData) {
            removeImage();
        }
        
    } catch (error) {
        console.error('Error:', error);
        typingIndicator.classList.add('hidden');
        appendMessage({
            text: 'Sorry, I encountered an error processing your request.',
            type: 'error'
        });
    }
}

function appendMessage(message, isUser = false) {
    // ... existing code ...

    // Add results if present
    if (message.results && message.results.length > 0) {
        const imageGrid = document.createElement('div');
        imageGrid.className = 'chat-image-grid mt-4';
        message.results.forEach(result => {
            if (!result.filename) return;
            
            const imgWrapper = document.createElement('div');
            imgWrapper.className = 'chat-image-wrapper';
            imgWrapper.innerHTML = `
                <img src="/images/${result.filename}" 
                     alt="${result.metadata?.description || 'Search result'}"
                     onerror="this.src='/static/images/placeholder.jpg'">
                <div class="chat-image-overlay">
                    <p class="text-white text-sm">${result.metadata?.description || ''}</p>
                </div>
            `;
            imgWrapper.onclick = () => showImageDetails(result.filename);
            imageGrid.appendChild(imgWrapper);
        });
        bubble.appendChild(imageGrid);
    }

    // ... rest of the existing code ...
}

async function processFiles(files) {
    const totalFiles = files.length;
    let successCount = 0;
    const statusText = document.getElementById('upload-status');
    const progressBar = document.getElementById('upload-progress');

    for (let i = 0; i < files.length; i++) {
        try {
            const formData = new FormData();
            formData.append('file', files[i]);
            // Add other form data...

            const response = await fetch('/api/v1/upload/file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                showError(errorData.detail || 'Error uploading file');
                continue;
            }

            successCount++;
            const progress = ((i + 1) / totalFiles) * 100;
            progressBar.style.width = `${progress}%`;
            statusText.textContent = `Uploaded ${i + 1} of ${totalFiles} files...`;

        } catch (error) {
            console.error(`Error uploading file ${i + 1}:`, error);
            showError(`Error uploading ${files[i].name}`);
        }
    }

    return successCount;
}
