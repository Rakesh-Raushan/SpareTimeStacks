
// Global variables to store current state
let currentFile = null;
let currentDataType = null;

// Augmentation options for different data types
const augmentationOptions = {
    text: [
        'Synonym Replacement',
        'Back Translation',
        'Random Insertion',
        'Random Deletion'
    ],
    image: [
        'Rotation',
        'Flip',
        'Color Adjustment',
        'Noise Addition'
    ],
    audio: [
        'Add Noise',
        'Speed Change'
    ],
    '3d-image': [
        'Rotation',
        'Vertex Displacement',
    ]
};

// DOM Elements
const dataTypeSelect = document.getElementById('dataType');
const fileInput = document.getElementById('fileInput');
const viewRawBtn = document.getElementById('viewRawBtn');
const preprocessBtn = document.getElementById('preprocessBtn');
const augmentBtn = document.getElementById('augmentBtn');
const augmentationOptionsDiv = document.getElementById('augmentationOptions');

// Event Listeners
dataTypeSelect.addEventListener('change', handleDataTypeChange);
fileInput.addEventListener('change', handleFileUpload);
viewRawBtn.addEventListener('click', viewRawData);
preprocessBtn.addEventListener('click', preprocessData);
augmentBtn.addEventListener('click', augmentData);

function handleDataTypeChange(event) {
    currentDataType = event.target.value;
    if (currentDataType) {
        fileInput.disabled = false;
        updateFileInputAccept();
        updateAugmentationOptions();
    } else {
        fileInput.disabled = true;
    }
    resetButtons();
}

function updateFileInputAccept() {
    const acceptMap = {
        'text': '.txt,.csv',
        'image': '.png,.jpg,.jpeg',
        'audio': '.mp3,.wav',
        '3d-image': '.off, .stl'
    };
    fileInput.accept = acceptMap[currentDataType] || '';
}

function updateAugmentationOptions() {
    augmentationOptionsDiv.innerHTML = '';
    if (currentDataType) {
        augmentationOptions[currentDataType].forEach(option => {
            const div = document.createElement('div');
            div.className = 'checkbox-group';
            div.innerHTML = `
                <input type="checkbox" id="${option}" name="augmentation" value="${option}">
                <label for="${option}">${option}</label>
            `;
            augmentationOptionsDiv.appendChild(div);
        });
    }
    
    // Add event listeners to checkboxes
    const checkboxes = document.querySelectorAll('input[name="augmentation"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateAugmentButton);
    });
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', currentDataType);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        currentFile = data.filename;
        viewRawBtn.disabled = false;
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
    }
}

async function viewRawData() {
    try {
        const response = await fetch(`/api/view-raw/${currentDataType}/${currentFile}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const rawDataContent = document.getElementById('rawDataContent');
        rawDataContent.innerHTML = ''; // Clear previous content

        if (currentDataType === 'text') {
            const data = await response.json();
            rawDataContent.textContent = JSON.stringify(data.sample, null, 2);
        } else if (['image'].includes(currentDataType)) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;
            imgElement.alt = currentFile;
            rawDataContent.appendChild(imgElement);
        } else if (['audio'].includes(currentDataType)) {
            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            const audioElement = document.createElement('audio');
            audioElement.controls = true;
            audioElement.src = audioUrl;
            rawDataContent.appendChild(audioElement);
        } else if (currentDataType === '3d-image') {
            const response = await fetch(`/api/view-raw/${currentDataType}/${currentFile}`);
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            console.log("vertices", data.vertices)
            console.log("faces", data.faces)
            renderThreeDModel(data.vertices, data.faces, 'rawDataContent');
        } else {
            rawDataContent.textContent = `Unsupported data type: ${currentDataType}`;
        }

        preprocessBtn.disabled = false;
    } catch (error) {
        console.error('Error viewing raw data:', error);
        alert('Error viewing raw data');
    }
}
        
async function preprocessData() {
    try {
        const response = await fetch(`/api/preprocess/${currentDataType}/${currentFile}`);
        console.log("preprocess response acha:", response);
        if (!response.ok) {
            throw new Error('Failed to preprocess data');
        }

        const preprocessedDataContent = document.getElementById('preprocessedDataContent');
        preprocessedDataContent.innerHTML = '';  // Clear previous content
        console.log("I am here in JS ")
        if (currentDataType === 'image') {
            // For image data type, display the image directly
            const img = document.createElement('img');
            img.src = response.url;  // The URL for the image file
            console.log("img src in the preprocess", img.src)
            img.alt = 'Preprocessed Image';
            preprocessedDataContent.appendChild(img);
        } else if (currentDataType === 'audio') {
            // Display preprocessed audio spectrogram (as an image)
            const blob = await response.blob();
            const audioImageUrl = URL.createObjectURL(blob);
            const img = document.createElement('img');
            img.src = audioImageUrl;
            img.alt = 'Preprocessed Audio Spectrogram';
            preprocessedDataContent.appendChild(img);
        } else if (currentDataType === '3d-image') {
            console.log("I am inside 3d bloc")
            // const response = await fetch(`/api/view-raw/${currentDataType}/${currentFile}`);
            const response = await fetch(`/api/preprocess/${currentDataType}/${currentFile}`);
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            console.log("vertices", data.vertices)
            console.log("faces", data.faces)
            renderThreeDModel(data.vertices, data.faces, 'preprocessedDataContent');
            } 
        else {
            // For other data types (e.g., text), display the preprocessed data as JSON
            const data = await response.json();
            const preElement = document.createElement('pre');
            preElement.textContent = JSON.stringify(data.preprocessed, null, 2);
            preprocessedDataContent.appendChild(preElement);
        }

        augmentBtn.disabled = false;
    } catch (error) {
        console.error('Error preprocessing data:', error);
        alert('Error preprocessing data');
    }
}


async function augmentData() {
    const selectedOptions = Array.from(document.querySelectorAll('input[name="augmentation"]:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedOptions.length === 0) {
        alert('Please select at least one augmentation option');
        return;
    }
    console.log("I am here in JS")

    const formData = new FormData();
    formData.append('augmentation_options', JSON.stringify(selectedOptions));
    console.log("Selected Augmentation Options:", selectedOptions);

    try {
        const response = await fetch(`/api/augment/${currentDataType}/${currentFile}`, {
            method: 'POST',
            body: formData
        });
        console.log("I have moved further here in JS")
        // const data = await response.json();
        
        // console.log("responseBlob:", blob);
        // console.log("currentFile:", currentFile);
        
        // console.log("URLfinally:", augmentedImageUrl);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error occurred');
        }

        // console.log("data:", data);

        const augmentedDataContent = document.getElementById('augmentedDataContent');

        // Clear previous content
        augmentedDataContent.innerHTML = '';

        if (currentDataType === 'image') {
            // If the data type is 'image', handle it as an image file
            // const augmentedImageUrl = data.augmented_image_url; // URL to the augmented image
            const blob = await response.blob();
            const augmentedImageUrl = URL.createObjectURL(blob);
            // Create an <img> element to display the augmented image
            const augmentedImageElement = document.createElement('img');
            // augmentedImageElement.src = augmentedImageUrl;
            augmentedImageElement.src = augmentedImageUrl;
            console.log("img src in the aug", augmentedImageElement.src)

            // Optional: Set the image's alt text for accessibility
            augmentedImageElement.alt = "Augmented Image";

            // Append the augmented image to the page
            augmentedDataContent.appendChild(augmentedImageElement);
            console.log("I have worked till end in JS", augmentedImageElement.src)
        } 
        if (currentDataType === 'audio') {
            // If the data type is 'audio', handle it as an audio file
            const blob = await response.blob();
            const augmentedAudioUrl = URL.createObjectURL(blob);
        
            // Create an <audio> element to play the augmented audio
            const augmentedAudioElement = document.createElement('audio');
            augmentedAudioElement.controls = true; // Add controls for playback
            augmentedAudioElement.src = augmentedAudioUrl;
        
            console.log("Audio src in the augmentation:", augmentedAudioElement.src);
        
            // Optional: Set the audio's title for accessibility
            augmentedAudioElement.title = "Augmented Audio";
        
            // Append the augmented audio to the page
            augmentedDataContent.appendChild(augmentedAudioElement);
        }
        if (currentDataType === 'text') {
            console.log("I am inside text bloc")
            const data = await response.json();
            console.log("I am after json  bloc")
            // If the data type is 'text', display the augmented text
            const preElement = document.createElement('pre');
            preElement.textContent = JSON.stringify(data.augmented, null, 2);

            // Append the augmented text to the page
            augmentedDataContent.appendChild(preElement);
        }
        if (currentDataType === '3d-image') {
            console.log("I am inside 3d aug bloc")
            // const data = await response.json();
            console.log("I am after json  bloc")
            // If the data type is 'text', display the augmented text
            // const response = await fetch(`/api/preprocess/${currentDataType}/${currentFile}`);
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            console.log("vertices", data.vertices)
            console.log("faces", data.faces)
            renderThreeDModel(data.vertices, data.faces, 'augmentedDataContent');
        }
    } catch (error) {
        console.error('Error augmenting data:', error);
        alert(`Error augmenting data: ${error.message}`);
    }
}



function updateAugmentButton() {
    const checkboxes = document.querySelectorAll('input[name="augmentation"]:checked');
    augmentBtn.disabled = checkboxes.length === 0;
}

function resetButtons() {
    viewRawBtn.disabled = true;
    preprocessBtn.disabled = true;
    augmentBtn.disabled = true;
    
    document.getElementById('rawDataContent').textContent = '';
    document.getElementById('preprocessedDataContent').textContent = '';
    document.getElementById('augmentedDataContent').textContent = '';
}

async function renderThreeDModel(vertices, faces, uielement) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    const rawDataContent = document.getElementById(uielement);
    rawDataContent.innerHTML = ''; // Clear previous content
    rawDataContent.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    geometry.setIndex(faces);

    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
    const model = new THREE.Mesh(geometry, material);
    scene.add(model);

    camera.position.z = 5;

    // Add event listeners for camera controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    animate();
}

// async function renderThreeDModel(vertices, faces) {
//     // Validate input data
//     if (!vertices || !faces || vertices.length === 0 || faces.length === 0) {
//         console.error('Invalid model data');
//         return;
//     }

//     const scene = new THREE.Scene();
//     const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
//     const renderer = new THREE.WebGLRenderer({ antialias: true });
//     renderer.setSize(window.innerWidth, window.innerHeight);
//     renderer.setClearColor(0xffffff, 1); // White background for better visibility
    
//     const rawDataContent = document.getElementById('rawDataContent');
//     rawDataContent.innerHTML = ''; // Clear previous content
//     rawDataContent.appendChild(renderer.domElement);

//     try {
//         const geometry = new THREE.BufferGeometry();
        
//         // Ensure vertices is a Float32Array
//         const verticesArray = vertices instanceof Float32Array ? vertices : new Float32Array(vertices);
//         geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
//         geometry.setIndex(faces);
        
//         // Compute vertex normals for better rendering
//         geometry.computeVertexNormals();

//         const material = new THREE.MeshPhongMaterial({ 
//             color: 0x00ff00,
//             wireframe: false,
//             side: THREE.DoubleSide
//         });
        
//         const model = new THREE.Mesh(geometry, material);
//         scene.add(model);

//         // Add lights
//         const ambientLight = new THREE.AmbientLight(0x404040);
//         scene.add(ambientLight);
//         const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
//         directionalLight.position.set(1, 1, 1);
//         scene.add(directionalLight);

//         // Position camera
//         camera.position.z = 2; // Closer since model is normalized to unit sphere

//         // Add orbit controls
//         const controls = new THREE.OrbitControls(camera, renderer.domElement);
//         controls.enableDamping = true; // Smooth camera movement
//         controls.dampingFactor = 0.05;
//         controls.screenSpacePanning = false;
//         controls.minDistance = 1;
//         controls.maxDistance = 10;
//         controls.update();

//         function animate() {
//             requestAnimationFrame(animate);
//             controls.update();
//             renderer.render(scene, camera);
//         }

//         animate();

//         // Handle window resize
//         window.addEventListener('resize', () => {
//             camera.aspect = window.innerWidth / window.innerHeight;
//             camera.updateProjectionMatrix();
//             renderer.setSize(window.innerWidth, window.innerHeight);
//         });

//     } catch (error) {
//         console.error('Error rendering 3D model:', error);
//         rawDataContent.innerHTML = 'Error rendering 3D model';
//     }
// }