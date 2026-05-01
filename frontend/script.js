document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const BASE_URL = "https://bis-smartstandards-ai-engine-2.onrender.com";
    const fileInput = document.getElementById('file-upload');
    const fileBtn = document.getElementById('btn-file-upload');
    const dropZone = document.getElementById('drop-zone');
    const fileNameDisplay = document.getElementById('file-name');

    const loadingState = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    const resultsList = document.getElementById('results-list');
    const responseTimeDisplay = document.getElementById('response-time');
    const downloadReportBtn = document.getElementById('btn-download-report');

    let selectedFile = null;
    let currentStandards = [];
    let currentMetadata = {};

    // --- Workflow: File Upload ---

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        handleFileSelection(files);
    });

    // Click Upload
    fileInput.addEventListener('change', (e) => {
        handleFileSelection(e.target.files);
    });

    function handleFileSelection(files) {
        if (files.length === 0) return;

        const file = files[0];

        // Validate type
        const validTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pdf', '.docx'
        ];

        if (!file.name.toLowerCase().endsWith('.pdf') && !file.name.toLowerCase().endsWith('.docx')) {
            showError("Only PDF and DOCX files are supported.");
            selectedFile = null;
            fileNameDisplay.textContent = "No file chosen";
            fileBtn.disabled = true;
            return;
        }

        // Validate size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError("File exceeds the 10MB limit.");
            selectedFile = null;
            fileNameDisplay.textContent = "No file chosen";
            fileBtn.disabled = true;
            return;
        }

        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.style.color = "var(--primary)";
        fileBtn.disabled = false;
        hideError();
    }

    fileBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        showLoading();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Unable to read the document. Please upload a valid PDF or Word file.");
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            showError(error.message);
        }
    });

    // --- Workflow 2: Direct Form Input ---
    const submitFormBtn = document.getElementById('btn-submit-form');

    if (submitFormBtn) {
        submitFormBtn.addEventListener('click', async () => {
            const pName = document.getElementById('inp-product-name').value.trim();
            const pCat = document.getElementById('inp-product-category').value.trim();
            const mfg = document.getElementById('inp-manufacturer').value.trim();
            const desc = document.getElementById('inp-description').value.trim();

            if (!desc && !pName && !pCat) {
                showError("Please fill in at least one field to analyze.");
                return;
            }

            const combinedText = `Product Name: ${pName}\nProduct Category: ${pCat}\nManufacturer Name: ${mfg}\nPRODUCT DESCRIPTION: ${desc}`;

            showLoading();
            submitFormBtn.disabled = true;

            try {
                const response = await fetch(`${BASE_URL}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: combinedText })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Analysis failed');
                }

                const data = await response.json();
                displayResults(data);

                // Clear the file selection if there was any
                selectedFile = null;
                if (fileNameDisplay) {
                    fileNameDisplay.textContent = "No file chosen";
                }

            } catch (error) {
                showError(error.message);
            } finally {
                submitFormBtn.disabled = false;
            }
        });
    }

    // --- Report Generation ---
    downloadReportBtn.addEventListener('click', async () => {
        if (currentStandards.length === 0) return;

        const originalText = downloadReportBtn.innerHTML;
        downloadReportBtn.innerHTML = 'Generating PDF... <div class="spinner" style="width: 16px; height: 16px; border-width: 2px; margin: 0 0 0 8px;"></div>';
        downloadReportBtn.disabled = true;

        try {
            const response = await fetch(`${BASE_URL}/generate-report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query_text: "Analyzed from uploaded document",
                    standards: currentStandards,
                    metadata: currentMetadata
                })
            });

            if (!response.ok) throw new Error("Failed to generate report");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            // Extract filename from Content-Disposition if present
            let filename = "BIS_SmartStandards_Report.pdf";
            const disposition = response.headers.get('content-disposition');
            if (disposition && disposition.indexOf('filename=') !== -1) {
                const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }

            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();

            window.URL.revokeObjectURL(url);
        } catch (error) {
            showError("Could not generate report: " + error.message);
        } finally {
            downloadReportBtn.innerHTML = originalText;
            downloadReportBtn.disabled = false;
        }
    });

    // --- UI Helpers ---

    function showLoading() {
        loadingState.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        fileBtn.disabled = true;
    }

    function hideLoading() {
        loadingState.classList.add('hidden');
        fileBtn.disabled = selectedFile ? false : true;
    }

    function showError(msg) {
        hideLoading();
        if (errorMessage) {
            errorMessage.textContent = msg;
            errorMessage.classList.remove('hidden');
        } else {
            alert(msg);
        }
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }

    function displayResults(data) {
        hideLoading();

        const standards = data.recommended_standards || [];

        if (standards.length === 0 || standards[0].standard === "No Match" || standards[0].standard === "No standard found") {
            showError("No matching BIS standard found. Please provide more details.");
            currentStandards = [];
            currentMetadata = {};
            return;
        }

        currentStandards = standards;
        currentMetadata = data.metadata || {};

        if (responseTimeDisplay) {
            responseTimeDisplay.textContent = `Response Time: ${data.response_time}s`;
        }
        if (resultsList) {
            resultsList.innerHTML = '';
        }

        standards.forEach((std, index) => {
            const confPercent = Math.round((std.confidence || 0.8) * 100);
            const confColor = confPercent > 80 ? 'var(--success)' : (confPercent > 50 ? '#F59E0B' : 'var(--error)');

            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.1}s`;

            card.innerHTML = `
                <div class="result-header">
                    <span class="result-standard">${std.standard}</span>
                    <span class="confidence-badge" style="color: ${confColor}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        ${confPercent}% Match
                    </span>
                </div>
                <div class="result-title">${std.title}</div>
                <div class="result-reason">
                    <strong>Reason:</strong> ${std.reason}
                </div>
            `;

            resultsList.appendChild(card);
        });

        resultsContainer.classList.remove('hidden');

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Expose globally for the Chatbot in translations.js
    window.displayResultsGlobally = displayResults;
});
