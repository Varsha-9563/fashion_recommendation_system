const API_BASE_URL = 'http://localhost:5001/api';
let currentUploadedImage = null;
let currentRandomCount = 8;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Fashion AI System...');
    checkSystemStatus();
    loadRandomProducts();
    loadRandomForIdSearch();
    setupEventListeners();
});

function setupEventListeners() {
    // File upload handling
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    // Click to upload
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== uploadArea && !uploadArea.contains(e.target)) return;
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            handleImageUpload(this.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Enter key support for search
    document.getElementById('productIdInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') getRecommendationsById();
    });
    
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') searchProducts();
    });
}

async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateSystemStatus('connected', `● CNN System Connected (${data.products_count} products)`);
            console.log('✅ Backend connected:', data);
        } else {
            updateSystemStatus('disconnected', '● CNN System Disconnected');
        }
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
        updateSystemStatus('disconnected', '● CNN System Disconnected');
    }
}

function updateSystemStatus(status, message) {
    const statusElement = document.getElementById('systemStatus');
    statusElement.className = `status-badge status-${status}`;
    statusElement.textContent = message;
}

function switchTab(tabName) {
    console.log('📑 Switching to tab:', tabName);
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Activate selected tab button
    const tabButtons = document.querySelectorAll('.tab');
    tabButtons.forEach(button => {
        if (button.textContent.includes(getTabIcon(tabName))) {
            button.classList.add('active');
        }
    });
    
    // Load content based on tab
    if (tabName === 'explore') {
        loadRandomProducts();
    } else if (tabName === 'id-search') {
        loadRandomForIdSearch();
    }
}

function getTabIcon(tabName) {
    const icons = {
        'image-search': '🖼️',
        'id-search': '🔍',
        'explore': '🧭'
    };
    return icons[tabName] || '';
}

function handleImageUpload(file) {
    if (!file.type.match('image.*')) {
        alert('Please upload an image file (JPEG, PNG, etc.)');
        return;
    }

    console.log('📤 Uploading image:', file.name);

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImage = document.getElementById('previewImage');
        previewImage.src = e.target.result;
        document.getElementById('uploadPreview').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
        
        currentUploadedImage = file;
        document.getElementById('analyzeBtn').disabled = false;
        console.log('✅ Image preview loaded');
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    console.log('🗑️ Clearing upload');
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInput').value = '';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('imageResults').style.display = 'none';
    currentUploadedImage = null;
}

async function getImageRecommendations() {
    if (!currentUploadedImage) {
        alert('Please upload an image first');
        return;
    }

    console.log('🔍 Getting recommendations for uploaded image...');
    showLoading('Analyzing image with CNN...', 'Extracting visual features using ResNet50');
    
    try {
        const formData = new FormData();
        formData.append('file', currentUploadedImage);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get recommendations');
        }

        const data = await response.json();
        console.log('✅ Received recommendations:', data);
        displayImageResults(data);
        
    } catch (error) {
        console.error('❌ Error:', error);
        alert('Error getting recommendations: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function getRecommendationsById() {
    const productId = document.getElementById('productIdInput').value.trim();
    
    if (!productId) {
        alert('Please enter a product ID');
        return;
    }

    console.log('🔍 Getting recommendations for product ID:', productId);
    showLoading('Finding similar items...', 'Searching database for matches');
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend/${productId}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Product not found');
        }

        const data = await response.json();
        console.log('✅ Received recommendations:', data);
        displayIdResults(data);
        
    } catch (error) {
        console.error('❌ Error:', error);
        alert('Error getting recommendations: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function searchProducts() {
    const query = document.getElementById('searchInput').value.trim();
    const category = document.getElementById('categorySelect').value;
    
    if (!query && !category) {
        alert('Please enter a search query or select a category');
        return;
    }

    console.log('🔍 Searching products:', { query, category });
    showLoading('Searching products...', 'Searching fashion database');
    
    try {
        const params = new URLSearchParams();
        if (query) params.append('q', query);
        if (category) params.append('category', category);

        const response = await fetch(`${API_BASE_URL}/search?${params}`);
        
        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        console.log('✅ Search results:', data);
        displaySearchResults(data);
        
    } catch (error) {
        console.error('❌ Error:', error);
        alert('Search failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function loadRandomProducts() {
    console.log('🎲 Loading random products for explore tab...');
    try {
        const response = await fetch(`${API_BASE_URL}/random?count=${currentRandomCount}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Loaded random products:', data.count);
            displayProducts(data.products || [], 'randomProductsGrid', false);
        }
    } catch (error) {
        console.error('❌ Error loading random products:', error);
    }
}

async function loadRandomForIdSearch() {
    console.log('🎲 Loading random products for ID search tab...');
    try {
        const response = await fetch(`${API_BASE_URL}/random?count=8`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Loaded products for clicking:', data.count);
            displayProducts(data.products || [], 'randomForIdSearchGrid', false);
        }
    } catch (error) {
        console.error('❌ Error loading products:', error);
    }
}

async function loadMoreRandom() {
    currentRandomCount += 8;
    console.log('📦 Loading more products, total:', currentRandomCount);
    await loadRandomProducts();
}

function displayImageResults(data) {
    const resultsSection = document.getElementById('imageResults');
    const statsElement = document.getElementById('resultsStats');
    
    const similarCount = data.similar_items?.length || 0;
    const outfitsCount = data.matching_outfits?.length || 0;
    
    console.log(`📊 Displaying results: ${similarCount} similar, ${outfitsCount} outfits`);
    
    // Update stats
    statsElement.textContent = `Found ${similarCount} similar items and ${outfitsCount} matching outfits`;
    
    // Display similar items
    displayProducts(data.similar_items || [], 'similarItemsGrid', true);
    
    // Display matching outfits
    displayProducts(data.matching_outfits || [], 'matchingOutfitsGrid', false);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayIdResults(data) {
    const resultsSection = document.getElementById('idResults');
    const queryInfoElement = document.getElementById('queryProductInfo');
    
    console.log('📊 Displaying ID search results...');
    
    // Display query product info
    if (data.queried_product) {
        const product = data.queried_product;
        queryInfoElement.innerHTML = `
            <h3>🔍 Selected Product</h3>
            <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                <img src="${getImageUrl(product)}" 
                     alt="${product.name}" 
                     style="width: 150px; height: 150px; object-fit: contain; border-radius: 10px; background: #f8f9fa;"
                     onerror="this.src='https://via.placeholder.com/150?text=No+Image'">
                <div>
                    <h4 style="margin-bottom: 10px;">${product.name || 'Unknown Product'}</h4>
                    <p style="margin: 5px 0;"><strong>ID:</strong> ${product.id}</p>
                    <p style="margin: 5px 0;"><strong>Category:</strong> ${product.category || 'N/A'}</p>
                    <p style="margin: 5px 0;"><strong>Type:</strong> ${product.article_type || 'N/A'}</p>
                    <p style="margin: 5px 0;"><strong>Color:</strong> ${product.color || 'N/A'}</p>
                </div>
            </div>
        `;
    }
    
    // Display similar items
    displayProducts(data.similar_items || [], 'idSimilarItemsGrid', true);
    
    // Display matching outfits
    displayProducts(data.matching_outfits || [], 'idMatchingOutfitsGrid', false);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displaySearchResults(data) {
    const resultsSection = document.getElementById('searchResults');
    const resultsGrid = document.getElementById('searchResultsGrid');
    
    console.log('📊 Displaying search results:', data.count);
    
    if (data.results && data.results.length > 0) {
        displayProducts(data.results, 'searchResultsGrid', false);
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
        resultsGrid.innerHTML = '<div class="no-results">No products found matching your search. Try different keywords.</div>';
        resultsSection.style.display = 'block';
    }
}

function displayProducts(products, containerId, showSimilarity = false) {
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error('❌ Container not found:', containerId);
        return;
    }
    
    if (products.length === 0) {
        container.innerHTML = '<div class="no-results">No products found.</div>';
        return;
    }
    
    console.log(`📦 Displaying ${products.length} products in ${containerId}`);
    
    container.innerHTML = products.map(product => {
        // Determine which score to show
        let scoreDisplay = '';
        if (showSimilarity) {
            if (product.ensemble_score !== undefined) {
                scoreDisplay = `<span class="similarity-badge">Match: ${Math.round(product.ensemble_score * 100)}%</span>`;
            } else if (product.similarity_score !== undefined) {
                scoreDisplay = `<span class="similarity-badge">Match: ${Math.round(product.similarity_score * 100)}%</span>`;
            }
        }
        
        return `
            <div class="product-card" onclick="getRecommendationsByIdFromCard(${product.id})">
                <img src="${getImageUrl(product)}" 
                     alt="${product.name || product.article_type || 'Product'}" 
                     class="product-image"
                     onerror="this.src='https://via.placeholder.com/250?text=No+Image'">
                <div class="product-info">
                    <div class="product-name">${product.name || product.article_type || 'Unknown Product'}</div>
                    <div class="product-details">${product.category || product.subCategory || 'N/A'}</div>
                    <div class="product-details">${product.article_type || 'N/A'}</div>
                    <div class="product-details" style="color: #666;">${product.color || product.baseColour || 'N/A'}</div>
                    <div class="product-meta">
                        <span class="category-badge">ID: ${product.id}</span>
                        ${scoreDisplay}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function getImageUrl(product) {
    // Priority: image_data (base64) > image_url > image_path > placeholder
    if (product.image_data) {
        return `data:image/jpeg;base64,${product.image_data}`;
    }
    if (product.image_url) {
        return product.image_url;
    }
    if (product.image_path) {
        // Try to construct URL from path
        return product.image_path;
    }
    return 'https://via.placeholder.com/250?text=No+Image';
}

function getRecommendationsByIdFromCard(productId) {
    console.log('🖱️ Product card clicked, ID:', productId);
    
    // Switch to ID search tab
    switchTab('id-search');
    
    // Set the product ID in the input
    document.getElementById('productIdInput').value = productId;
    
    // Scroll to top of page
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    // Get recommendations after a short delay to allow tab switch
    setTimeout(() => {
        getRecommendationsById();
    }, 300);
}

function showLoading(message = 'Processing...', subtext = 'Please wait...') {
    document.getElementById('loadingMessage').textContent = message;
    document.getElementById('loadingSubtext').textContent = subtext;
    document.getElementById('loadingOverlay').style.display = 'flex';
    console.log('⏳ Loading:', message);
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
    console.log('✅ Loading complete');
}

// Log system ready
console.log('✅ Fashion AI System Ready!');