class Translator {
    constructor() {
        this.currentLang = localStorage.getItem('language') || 'en';
        
        const langSelect = document.getElementById('language-selector');
        if (langSelect) {
            langSelect.value = this.currentLang;
            langSelect.addEventListener('change', (e) => {
                this.setLanguage(e.target.value);
            });
        }

        if (this.currentLang !== 'en') {
            setTimeout(() => this.translatePage(), 500);
        }
    }

    async setLanguage(lang) {
        this.currentLang = lang;
        localStorage.setItem('language', lang);
        
        if (lang === 'en') {
            location.reload(); 
            return;
        }
        
        await this.translatePage();
    }

    async translateText(text) {
        if (this.currentLang === 'en' || !text.trim()) return text;
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, target_lang: this.currentLang })
            });
            const data = await response.json();
            return data.translated_text;
        } catch (e) {
            return text;
        }
    }

    async translatePage() {
        const elements = document.querySelectorAll('h1, h2, h3, h4, p, label, button, .file-name, .upload-text, li, .badge, span[data-i18n]');
        
        const toTranslate = [];
        const elMap = [];

        for (const el of elements) {
            if (el.closest('svg') || el.id === 'language-selector') continue;

            if (!el.hasAttribute('data-en')) {
                let textToTranslate = "";
                for(let child of el.childNodes) {
                    if (child.nodeType === 3 && child.textContent.trim()) {
                        textToTranslate += child.textContent;
                    }
                }
                if (textToTranslate.trim()) {
                    el.setAttribute('data-en', textToTranslate.trim());
                }
            }
            
            if (el.hasAttribute('data-en')) {
                toTranslate.push(el.getAttribute('data-en'));
                elMap.push(el);
            }
        }
        
        const inputs = document.querySelectorAll('input[placeholder], textarea[placeholder]');
        for (const input of inputs) {
            if (!input.hasAttribute('data-en-placeholder')) {
                input.setAttribute('data-en-placeholder', input.placeholder);
            }
            toTranslate.push(input.getAttribute('data-en-placeholder'));
            elMap.push(input);
        }

        if (toTranslate.length === 0) return;

        document.body.style.opacity = 0.7;
        const joinedText = toTranslate.join(' \n\n ');
        const translatedText = await this.translateText(joinedText);
        const translatedArray = translatedText.split(/\n\s*\n/).map(s => s.trim());

        for (let i = 0; i < elMap.length; i++) {
            const trans = translatedArray[i] || toTranslate[i];
            const el = elMap[i];
            
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                el.placeholder = trans;
            } else {
                for(let child of el.childNodes) {
                    if (child.nodeType === 3 && child.textContent.trim()) {
                        child.textContent = trans;
                        break;
                    }
                }
            }
        }
        document.body.style.opacity = 1;
    }
}

window.appTranslator = new Translator();

// --- Chatbot Logic ---
document.addEventListener('DOMContentLoaded', () => {
    const chatToggle = document.getElementById('chatbot-toggle');
    const chatPanel = document.getElementById('chatbot-panel');
    const chatClose = document.getElementById('chatbot-close');
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chatbot-messages');
    const chatSuggestions = document.querySelectorAll('.chat-suggestion-btn');

    let chatState = 0; 
    let chatData = { pName: "", pCat: "", mfg: "", desc: "" };

    chatToggle.addEventListener('click', () => {
        chatPanel.classList.remove('hidden');
        chatToggle.style.display = 'none';
        
        // Translate welcome message if not already translated
        if (window.appTranslator.currentLang !== 'en') {
            const welcomeMsg = chatMessages.querySelector('.message-bubble');
            if (welcomeMsg && !welcomeMsg.hasAttribute('data-translated')) {
                window.appTranslator.translateText(welcomeMsg.textContent).then(res => {
                    welcomeMsg.textContent = res;
                    welcomeMsg.setAttribute('data-translated', 'true');
                });
            }
        }
    });

    chatClose.addEventListener('click', () => {
        chatPanel.classList.add('hidden');
        chatToggle.style.display = 'flex';
    });

    async function addMessage(text, isUser = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        let displayText = text;
        if (!isUser && window.appTranslator && window.appTranslator.currentLang !== 'en') {
            displayText = await window.appTranslator.translateText(text);
        }
        
        msgDiv.innerHTML = `<div class="message-bubble">${displayText}</div>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    chatSend.addEventListener('click', () => {
        const text = chatInput.value.trim();
        if (!text) return;
        
        addMessage(text, true);
        chatInput.value = '';
        handleUserMessage(text);
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') chatSend.click();
    });

    chatSuggestions.forEach(btn => {
        btn.addEventListener('click', () => {
            const originalText = btn.getAttribute('data-en') || btn.textContent;
            addMessage(originalText, true); // User sends english context internally
            handleUserMessage(btn.getAttribute('data-action') || originalText);
        });
    });

    async function handleUserMessage(msg) {
        const lowerMsg = msg.toLowerCase();
        
        if (msg === 'identify' || lowerMsg.includes('identify') || lowerMsg.includes('product') || lowerMsg.includes('standard')) {
            chatState = 1;
            await addMessage("Great! Let's find the standard. First, what is the name of your product? (e.g., Concrete Blocks)");
            return;
        }
        
        if (msg === 'form' || lowerMsg.includes('form') || lowerMsg.includes('upload')) {
            await addMessage("To get started, download the PDF template from the left side, fill in your product details, and upload it back here!");
            return;
        }
        
        if (msg === 'explain' || lowerMsg.includes('explain') || lowerMsg.includes('why')) {
            await addMessage("Our RAG engine matches the semantic keywords in your product description with the official BIS SP 21 corpus to find the most accurate standards mathematically.");
            return;
        }
        
        if (chatState === 1) {
            chatData.pName = msg;
            chatState = 2;
            await addMessage(`Got it. Your product is ${msg}. What is its category? (e.g., Building Materials)`);
        } else if (chatState === 2) {
            chatData.pCat = msg;
            chatState = 3;
            await addMessage(`Okay. Who is the manufacturer or company name?`);
        } else if (chatState === 3) {
            chatData.mfg = msg;
            chatState = 4;
            await addMessage(`Finally, please describe what it is made of and how it is used.`);
        } else if (chatState === 4) {
            chatData.desc = msg;
            chatState = 0;
            await addMessage(`Thanks! Analyzing your details now...`);
            
            const combinedText = `Product Name: ${chatData.pName}\nProduct Category: ${chatData.pCat}\nManufacturer Name: ${chatData.mfg}\nPRODUCT DESCRIPTION: ${chatData.desc}`;
            
            fetchPredict(combinedText);
        } else {
            await addMessage("I am the BIS Smart Assistant. I can help you identify standards for your product, explain recommendations, or help you fill out forms. Try clicking one of the suggestions below!");
        }
    }

    async function fetchPredict(text) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            
            if(window.displayResultsGlobally) {
                window.displayResultsGlobally(data);
                await addMessage("Analysis complete! I have generated your recommendations in the main dashboard.");
            } else {
                await addMessage("Analysis complete, but unable to display results on the main screen.");
            }
        } catch (e) {
            await addMessage("Sorry, an error occurred during analysis.");
        }
    }
});
