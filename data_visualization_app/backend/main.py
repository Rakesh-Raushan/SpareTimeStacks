from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import re, os, shutil
from backend.helpers import upload, preprocess#, augment_text
import random, json
from nltk.corpus import wordnet
from googletrans import Translator
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import TimeStretch, PitchShift, Resample, Vol
import logging
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation


app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# File type validation
ALLOWED_EXTENSIONS = {
    'text': {'.txt', '.csv'},
    'image': {'.png', '.jpeg', '.jpg'},
    'audio': {'.mp3', '.wav'},
    '3d-image': {'.off', '.stl'}
}

# Add route for serving index.html
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_file(file: UploadFile, data_type: str = Form(...)):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS[data_type]:
        return {"error": f"Invalid file type for {data_type} data"}
    
    # Save file to temporary location
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "path": temp_path}


@app.get("/api/view-raw/{data_type}/{filename}")
async def view_raw_data(data_type: str, filename: str):
    file_path = Path(f"temp/{filename}")
    
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Handle text files
    if data_type == "text":
        if file_path.suffix not in ['.txt', '.csv']:
            return JSONResponse({"error": "Unsupported text file format"}, status_code=400)
        with file_path.open("r") as f:
            sample = f.readlines()[:5]  # First 5 lines
        return {"sample": sample}

    # Handle image files
    elif data_type == "image":
        if file_path.suffix not in [".png", ".jpg", ".jpeg"]:
            return JSONResponse({"error": "Unsupported image format"}, status_code=400)
        return FileResponse(str(file_path), media_type=f"image/{file_path.suffix.lstrip('.')}")

    # Handle audio files (MP3)
    elif data_type == "audio":
        if file_path.suffix != ".mp3":
            return JSONResponse({"error": "Only MP3 audio files are supported"}, status_code=400)
        return FileResponse(str(file_path), media_type="audio/mpeg")

    # Handle 3D model files (OFF)
    # elif data_type == "3d-image":
    #     if file_path.suffix != ".off":
    #         return JSONResponse({"error": "Only OFF 3D model files are supported"}, status_code=400)
    #     return FileResponse(str(file_path), media_type="application/octet-stream")
    if data_type == "3d-image":
        if file_path.suffix != ".off":
            return JSONResponse({"error": "Only OFF 3D model files are supported"}, status_code=400)
        
        try:
            with open(str(file_path), "r") as f:
                lines = f.readlines()
            
            # Parse the OFF file format
            header_match = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", lines[1])
            if not header_match:
                return JSONResponse({"error": "Invalid OFF file format"}, status_code=400)
            
            num_vertices, num_faces, _ = [int(x) for x in header_match.groups()]
            vertices = []
            faces = []
            
            for i in range(2, 2 + num_vertices):
                vertex_data = [float(c) for c in lines[i].split()]
                vertices.extend(vertex_data)
            
            for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
                face_data = [int(x) for x in lines[i].split()]
                for j in range(1, face_data[0] + 1):
                    faces.append(face_data[j])
            
            return {"vertices": vertices, "faces": faces}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    # Other data type handlers...

    # Default for unsupported types
    return {"message": f"Raw data view for {data_type} not implemented"}

@app.get("/api/preprocess/{data_type}/{filename}")
async def preprocess_data(data_type: str, filename: str):
    file_path = Path(f"temp/{filename}")
    if not file_path.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    # Implement preprocessing logic based on data type
    if data_type == "text":
        if file_path.suffix not in ['.txt', '.csv']:
            return JSONResponse({"error": "Unsupported text file format"}, status_code=400)
        with file_path.open("r") as f:
            sample = f.readlines()[:5]  # First 5 lines
            sample_str = " ".join(sample)
            preprocessed_data = preprocess(sample_str, data_type)  # Pass file_type to your preprocessing function
        return {"preprocessed": preprocessed_data}
    
    if data_type == "image":
        try:
            image = Image.open(file_path)
            
            # Basic preprocessing
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize to 128x128
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            ])
            preprocessed_image = transform(image)
            # Save the preprocessed image with a new filename
            preprocessed_file_path = file_path.with_stem(f"{file_path.stem}_preprocessed")
            preprocessed_image.save(preprocessed_file_path)
            
            # Return the preprocessed image as a file response
            return FileResponse(
                str(preprocessed_file_path), 
                media_type=f"image/{preprocessed_file_path.suffix.lstrip('.')}",
                filename=preprocessed_file_path.name
            )
        
        except Exception as e:
            return JSONResponse({"error": f"Failed to preprocess image: {str(e)}"}, status_code=500)
        
    if data_type == "audio":
        if file_path.suffix != ".mp3":
            return JSONResponse({"error": "Only MP3 audio files are supported"}, status_code=400)
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=None)  # Use the original sampling rate
            
            # Example preprocessing: compute a Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Save the spectrogram as an image
            spectrogram_path = file_path.with_stem(f"{file_path.stem}_spectrogram").with_suffix(".png")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(spectrogram_path)
            plt.close()

            return FileResponse(
                str(spectrogram_path), 
                media_type="image/png", 
                filename=spectrogram_path.name
            )
        except Exception as e:
            return JSONResponse({"error": f"Failed to preprocess audio: {str(e)}"}, status_code=500)
        
    if data_type == "3d-image":
        if file_path.suffix != ".off":
            return JSONResponse({"error": "Only OFF 3D model files are supported"}, status_code=400)
    
        try:
            with open(str(file_path), "r") as f:
                lines = f.readlines()
            
            # Parse the OFF file format
            if not lines[0].strip().upper() == "OFF":
                return JSONResponse({"error": "Invalid OFF file format"}, status_code=400)
                
            header_match = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", lines[1])
            if not header_match:
                return JSONResponse({"error": "Invalid OFF file format"}, status_code=400)
            
            num_vertices, num_faces, _ = [int(x) for x in header_match.groups()]
            vertices = []
            faces = []
            
            # Parse vertices
            for i in range(2, 2 + num_vertices):
                vertex_data = [float(c) for c in lines[i].split()]
                if len(vertex_data) != 3:
                    return JSONResponse({"error": f"Invalid vertex data at line {i}"}, status_code=400)
                vertices.extend(vertex_data)
            
            # Parse faces
            for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
                face_data = [int(x) for x in lines[i].split()]
                if face_data[0] != 3:  # Ensure triangular faces
                    return JSONResponse({"error": "Only triangular faces are supported"}, status_code=400)
                faces.extend(face_data[1:4])  # Only take the vertex indices

            # Convert vertices to numpy array and reshape
            vertices_array = np.array(vertices, dtype=np.float32).reshape(-1, 3)
            
            # Apply stretching transformation
            stretch_factors = (2.0,1.0,1.0)
            stretch_matrix = np.array([
                [stretch_factors[0], 0, 0],
                [0, stretch_factors[1], 0],
                [0, 0, stretch_factors[2]]
            ], dtype=np.float32)
            
            # Apply stretching
            stretched_vertices = vertices_array @ stretch_matrix
            
            # Center the model
            centroid = np.mean(stretched_vertices, axis=0)
            centered_vertices = stretched_vertices - centroid
            
            # Scale to reasonable size
            max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
            if max_distance == 0:
                return JSONResponse({"error": "Invalid model: all vertices are at the same point"}, status_code=400)
            
            normalized_vertices = (centered_vertices / max_distance) * 2  # Scale to fit in a 4x4x4 cube
            
            # Convert to flat array for Three.js
            vertices_flat = normalized_vertices.flatten().tolist()
            
            # Validate data
            if len(vertices_flat) % 3 != 0:
                return JSONResponse({"error": "Invalid vertex data"}, status_code=400)
            if len(faces) % 3 != 0:
                return JSONResponse({"error": "Invalid face data"}, status_code=400)
                
            # Check for NaN values
            if np.any(np.isnan(vertices_flat)):
                return JSONResponse({"error": "Invalid vertex data: NaN values detected"}, status_code=400)
            
            return JSONResponse({
                "vertices": vertices_flat,  # Flat array of floats
                "faces": faces  # Array of indices
            }, status_code=200)

        except Exception as e:
            logger.error(f"Error preprocessing {data_type} file {filename}: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to preprocess file: {str(e)}"}, 
                status_code=500
            )
        
    return JSONResponse({"error": f"Unsupported data type: {data_type}"}, status_code=400)


@app.post("/api/augment/{data_type}/{filename}")
async def augment_data(data_type: str, filename: str, augmentation_options: list = Form(...)):
    file_path = Path(f"temp/{filename}")

    if not file_path.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    if data_type == "text":
        with file_path.open("r") as f:
            content = f.read()
        augmented_content = augment_text(content, augmentation_options)
        return JSONResponse({"augmented": augmented_content}, status_code=200)

    if data_type == "image":
        print("I am here in the image part of main api")
        try:
            # Open the image
            image = Image.open(file_path)

            # Parse the augmentation options from the form data
            options = json.loads(augmentation_options[0])
            print("Received Augmentation Options:", options)
            transform_list = []
            
            # Apply augmentations based on selected options
            if "Rotation" in options:
                transform_list.append(transforms.RandomRotation(degrees=30))
            if "Flip" in options:
                print("I am inside Flip")
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if "Color Adjustment" in options:
                transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            if "Noise Addition" in options:
                # Custom noise addition using random noise
                transform_list.append(transforms.Lambda(lambda img: add_noise(img)))

            # Apply the transformations
            transform = transforms.Compose(transform_list)
            augmented_image = transform(image)
            
            # Assuming the augmented image is generated (in PIL Image format)
            #augmented_image_pil = augmented_image if isinstance(augmented_image, Image.Image) else transforms.ToPILImage()(augmented_image)

            # Save the augmented image to a new file
            augmented_image_path = file_path.with_stem(f"{file_path.stem}_augmented")
            # augmented_image_path = file_path.with_name(f"{file_path.stem}_augmented{file_path.suffix}")
            augmented_image.save(augmented_image_path)

            # return JSONResponse(content={"augmented_image_url": f"/static/{augmented_image_path.name}"})
            # Return the augmented image as a file response
            return FileResponse(
                str(augmented_image_path),
                media_type=f"image/{augmented_image_path.suffix.lstrip('.')}",
                filename=augmented_image_path.name
            )
        
        except Exception as e:
            return JSONResponse({"error": f"Failed to process image: {str(e)}"}, status_code=500)
        
    if data_type == "audio":
        try:
            waveform, original_sample_rate = torchaudio.load(file_path)
            
            # Parse the augmentation options from the form data
            options = json.loads(augmentation_options[0])
            print("Received Augmentation Options:", options)

            # TimeStretch requires a spectrogram (complex STFT representation)
            spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, power=None)
            spectrogram = spectrogram_transform(waveform)

            # if "Time Stretch" in options:
            #     # Convert to spectrogram
            #     spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, power=1)
            #     spectrogram = spectrogram_transform(waveform)

            #     # Apply Time Stretch
            #     time_stretch = torchaudio.transforms.TimeStretch(hop_length=512)
            #     stretched_spectrogram = time_stretch(spectrogram, rate=1.2)

            #     # Convert back to waveform
            #     waveform = torchaudio.functional.griffin_lim(stretched_spectrogram, n_fft=2048, hop_length=512)

            # if "Pitch Shift" in options:
            #     pitch_shift = torchaudio.transforms.PitchShift(sample_rate=original_sample_rate, n_steps=2)
            #     waveform = pitch_shift(waveform.detach().clone())

            if "Add Noise" in options:
                noise = torch.randn_like(waveform) * 0.21  # Adding small Gaussian noise
                waveform = waveform + noise

            if "Speed Change" in options:
                new_sample_rate = int(original_sample_rate * 2.5)  # Change speed by 1.5x
                resample = Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
                waveform = resample(waveform)

            # Save the augmented audio
            augmented_audio_path = file_path.with_stem(f"{file_path.stem}_augmented")
            torchaudio.save(augmented_audio_path, waveform, sample_rate=original_sample_rate)

            return FileResponse(
                str(augmented_audio_path),
                media_type="audio/mpeg",
                filename=augmented_audio_path.name
            )

        except Exception as e:
            return JSONResponse({"error": f"Failed to process audio: {str(e)}"}, status_code=500)
        
    if data_type == "3d-image":
        if file_path.suffix != ".off":
            return JSONResponse({"error": "Only OFF 3D model files are supported"}, status_code=400)
    
        try:
            options = json.loads(augmentation_options[0])
            print("Received Augmentation Options:", options)
            # some hard coded values
            stretch_factors = (1.0, 1.0, 1.0)
            
            if 'Rotation' in options:
                rotation_angles=(45, 30, 0)
            else: rotation_angles=(0,0,0)
            if 'Vertex Displacement' in options:
                displacement_params={
                "type": "sine_wave",
                "amplitude": 0.2,
                "frequency": 2.0
            }
            else: displacement_params = None

            # Read and parse OFF file
            with open(str(file_path), "r") as f:
                lines = f.readlines()
            
            if not lines[0].strip().upper() == "OFF":
                return JSONResponse({"error": "Invalid OFF file format"}, status_code=400)
                
            header_match = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", lines[1])
            if not header_match:
                return JSONResponse({"error": "Invalid OFF file format"}, status_code=400)
            
            num_vertices, num_faces, _ = map(int, header_match.groups())
            vertices = []
            faces = []
            
            # Parse vertices
            for i in range(2, 2 + num_vertices):
                vertex_data = [float(c) for c in lines[i].split()]
                if len(vertex_data) != 3:
                    return JSONResponse({"error": f"Invalid vertex data at line {i}"}, status_code=400)
                vertices.extend(vertex_data)
            
            # Parse faces
            for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
                face_data = [int(x) for x in lines[i].split()]
                if face_data[0] != 3:
                    return JSONResponse({"error": "Only triangular faces are supported"}, status_code=400)
                faces.extend(face_data[1:4])
                
            # Convert vertices to numpy array and reshape
            vertices_array = np.array(vertices, dtype=np.float32).reshape(-1, 3)
            
            # Create transformation matrices
            stretch_matrix = np.diag(stretch_factors)
            rotation_matrix = create_rotation_matrix(rotation_angles)
            
            # Apply stretching
            transformed_vertices = vertices_array @ stretch_matrix
            
            # Apply rotation
            transformed_vertices = transformed_vertices @ rotation_matrix
            
            # Apply displacement if specified
            if displacement_params:
                displacement_func = create_displacement_function(
                    displacement_params.get('type', 'none'),
                    displacement_params.get('amplitude', 1.0),
                    displacement_params.get('frequency', 1.0),
                    displacement_params.get('seed')
                )
                transformed_vertices += displacement_func(transformed_vertices)
            
            # Center the model
            centroid = np.mean(transformed_vertices, axis=0)
            centered_vertices = transformed_vertices - centroid
            
            # Scale to reasonable size
            max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
            if max_distance == 0:
                return JSONResponse({"error": "Invalid model: all vertices are at the same point"}, status_code=400)
            
            normalized_vertices = (centered_vertices / max_distance) * 2
            
            # Convert to flat array for Three.js
            vertices_flat = normalized_vertices.flatten().tolist()
            
            # Validate data
            if len(vertices_flat) % 3 != 0:
                return JSONResponse({"error": "Invalid vertex data"}, status_code=400)
            if len(faces) % 3 != 0:
                return JSONResponse({"error": "Invalid face data"}, status_code=400)
            if np.any(np.isnan(vertices_flat)):
                return JSONResponse({"error": "Invalid vertex data: NaN values detected"}, status_code=400)
            
            return JSONResponse({
                "vertices": vertices_flat,
                "faces": faces
            }, status_code=200)
        except Exception as e:
            return JSONResponse({"error": f"Error processing file: {str(e)}"}, status_code=400)

    return JSONResponse({"error": f"Unsupported data type: {data_type}"}, status_code=400)
    


def add_noise(img: Image.Image, noise_factor=0.1):
    """ Adds random noise to the image """
    img_tensor = transforms.ToTensor()(img)
    noise = torch.randn_like(img_tensor) * noise_factor
    noisy_img_tensor = img_tensor + noise
    noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)  # Clamps the values to valid image range
    return transforms.ToPILImage()(noisy_img_tensor)


def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words[:]
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        synonym = random.choice(synonyms).lemmas()[0].name()
        new_words = [synonym if word == random_word else word for word in new_words]
        num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

def back_translation(text, src_lang="en", interim_lang="fr"):
    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=interim_lang).text
    back_translated = translator.translate(translated, src=interim_lang, dest=src_lang).text
    return back_translated

def random_insertion(text, n=2):
    words = text.split()
    for _ in range(n):
        new_word = random.choice([word for word in words if wordnet.synsets(word)])
        synonym = wordnet.synsets(new_word)[0].lemmas()[0].name()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, synonym)
    return " ".join(words)

def random_deletion(text, p=0.2):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return " ".join(new_words) if new_words else random.choice(words)

def augment_text(text, options):
    try:
        options = json.loads(options[0])
    except json.JSONDecodeError:
        return "Invalid augmentation options"
    augmented_dict = {}
    if "Synonym Replacement" in options:
        syn_rep_text = synonym_replacement(text)
        augmented_dict["After Synonym Replacement"]= syn_rep_text
    if "Back Translation" in options:
        back_trans_text = back_translation(text)
        augmented_dict["After Back Translation"]= back_trans_text
    if "Random Insertion" in options:
        rand_ins_text = random_insertion(text)
        augmented_dict["After Random Insertion"]= rand_ins_text
    if "Random Deletion" in options:
        rand_del_text = random_deletion(text)
        augmented_dict["After Random Deletion"]= rand_del_text
    return augmented_dict


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_off_file(file_path: Path):
    """
    Parse the OFF file format and return vertices and faces.
    """
    try:
        with open(str(file_path), "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
        
        # Check if file is empty
        if not lines:
            raise ValueError("Empty file")
            
        # Verify OFF header
        if not lines[0].strip().upper() == "OFF":
            raise ValueError("Missing OFF header")
            
        # Parse header numbers
        try:
            header_parts = lines[1].split()
            if len(header_parts) < 3:
                raise ValueError("Invalid header format")
            num_vertices, num_faces, num_edges = map(int, header_parts)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing header: {str(e)}")

        # Validate file has enough lines
        expected_lines = 2 + num_vertices + num_faces
        if len(lines) < expected_lines:
            raise ValueError(f"File has {len(lines)} lines, expected {expected_lines}")

        vertices = []
        faces = []

        # Parse vertices
        for i in range(2, 2 + num_vertices):
            try:
                vertex = [float(x) for x in lines[i].split()]
                if len(vertex) != 3:
                    raise ValueError(f"Invalid vertex format at line {i+1}")
                vertices.extend(vertex)
            except Exception as e:
                raise ValueError(f"Error parsing vertex at line {i+1}: {str(e)}")

        # Parse faces
        for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
            try:
                face_data = [int(x) for x in lines[i].split()]
                if face_data[0] != 3:  # Assuming triangular faces
                    raise ValueError(f"Non-triangular face at line {i+1}")
                faces.extend(face_data[1:4])  # Only take the three vertex indices
            except Exception as e:
                raise ValueError(f"Error parsing face at line {i+1}: {str(e)}")

        logger.debug(f"Successfully parsed {len(vertices)//3} vertices and {len(faces)//3} faces")
        return vertices, faces

    except Exception as e:
        logger.error(f"Error parsing OFF file: {str(e)}")
        raise

def normalize_vertices(vertices):
    """
    Normalize the 3D vertices to a unit sphere.
    """
    try:
        # Convert to numpy array for easier manipulation
        vertices_array = np.array(vertices).reshape(-1, 3)
        
        # Calculate centroid
        centroid = np.mean(vertices_array, axis=0)
        logger.debug(f"Centroid: {centroid}")
        
        # Center the vertices
        centered_vertices = vertices_array - centroid
        
        # Calculate scaling factor (maximum distance from center)
        max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
        logger.debug(f"Max distance from center: {max_distance}")
        
        if max_distance == 0:
            raise ValueError("All vertices are at the same point")
        
        # Scale to unit sphere
        normalized_vertices = centered_vertices / max_distance
        
        # Convert back to list format
        normalized_list = normalized_vertices.flatten().tolist()
        
        logger.debug(f"Normalized {len(normalized_list)//3} vertices")
        return normalized_list

    except Exception as e:
        logger.error(f"Error normalizing vertices: {str(e)}")
        raise

def save_off_file(file_path: Path, vertices, faces):
    """
    Save the preprocessed 3D data to an OFF file.
    """
    try:
        num_vertices = len(vertices) // 3
        num_faces = len(faces) // 3
        
        logger.debug(f"Saving file with {num_vertices} vertices and {num_faces} faces")
        
        with open(str(file_path), "w") as f:
            # Write header
            f.write("OFF\n")
            f.write(f"{num_vertices} {num_faces} 0\n")
            
            # Write vertices
            for i in range(0, len(vertices), 3):
                f.write(f"{vertices[i]:.6f} {vertices[i+1]:.6f} {vertices[i+2]:.6f}\n")
            
            # Write faces
            for i in range(0, len(faces), 3):
                f.write(f"3 {faces[i]} {faces[i+1]} {faces[i+2]}\n")

        logger.debug(f"Successfully saved to {file_path}")

    except Exception as e:
        logger.error(f"Error saving OFF file: {str(e)}")
        raise

def create_rotation_matrix(angles: Tuple[float, float, float]) -> np.ndarray:
    """
    Creates a 3D rotation matrix from Euler angles (in degrees).
    
    Args:
        angles: Tuple of (x, y, z) rotation angles in degrees
    
    Returns:
        3x3 rotation matrix
    """
    # Convert degrees to radians and create rotation matrix
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    return rotation.as_matrix()

def create_displacement_function(
    displacement_type: str,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    seed: Optional[int] = None
) -> callable:
    """
    Creates a vertex displacement function.
    
    Args:
        displacement_type: Type of displacement ('sine_wave', 'noise', 'random')
        amplitude: Maximum displacement magnitude
        frequency: Frequency of displacement pattern
        seed: Random seed for reproducible results
        
    Returns:
        Function that takes vertex coordinates and returns displacement vector
    """
    if seed is not None:
        np.random.seed(seed)
        
    if displacement_type == 'sine_wave':
        def displace(vertices):
            # Creates a sine wave pattern along y-axis
            return np.column_stack([
                np.zeros_like(vertices[:, 0]),
                amplitude * np.sin(frequency * vertices[:, 0]),
                np.zeros_like(vertices[:, 2])
            ])
            
    elif displacement_type == 'noise':
        # Create 3D Perlin-like noise
        noise_grid = np.random.rand(10, 10, 10) * 2 - 1
        def displace(vertices):
            # Map vertices to noise grid coordinates
            grid_coords = (vertices + 1) * 4.5  # Map [-1,1] to [0,9]
            grid_coords = np.clip(grid_coords, 0, 9)
            grid_x = grid_coords[:, 0].astype(int)
            grid_y = grid_coords[:, 1].astype(int)
            grid_z = grid_coords[:, 2].astype(int)
            return amplitude * np.column_stack([
                noise_grid[grid_x, grid_y, grid_z],
                noise_grid[grid_y, grid_z, grid_x],
                noise_grid[grid_z, grid_x, grid_y]
            ])
            
    elif displacement_type == 'random':
        def displace(vertices):
            return amplitude * (np.random.rand(len(vertices), 3) * 2 - 1)
            
    else:
        def displace(vertices):
            return np.zeros_like(vertices)
            
    return displace