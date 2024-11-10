from backend.utils import get_tokenizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.corpus import wordnet 
import random
from torchvision import transforms
from PIL import Image
import io
import numpy as np

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))
# text data functions

def upload_text(contents):
    data = contents.decode("utf-8").splitlines()  # Decode the contents
    sample_data = data[0]  # Get the first line as a sample
    return sample_data

def preprocess_text(text):
    # remove all "-" from text
    text =  text.replace("-"," ")
    # convert text to lower
    lowered_text = text.lower()
    # remove stop words
    tokeniser = get_tokenizer('basic_english')
    # creating tokens
    tokens = tokeniser(lowered_text)
    # removing stop words
    filtered_tokens = [t for t in tokens if t.lower() not in stop_words]
    # stemmisation
    stemmed_tokens = [stemmer.stem(tok) for tok in filtered_tokens]
    # Removing rare words
    freq_distribution = FreqDist(stemmed_tokens)
    common_tokens = [tok for tok in stemmed_tokens if freq_distribution[tok] > 2]

    preprocessed_dict = {"Tokenisation output": tokens,
                         "After stop word removal": filtered_tokens,
                         "After stemming": stemmed_tokens,
                         "After rare word removal": common_tokens}
    return preprocessed_dict


def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(text, n=2):
    
    words = text.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    print(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(words)
    new_sentence = ' '.join(new_words)

    return new_sentence

# def augment_text(text):
#     # replacement with synonyms
#     orig_text, synonymed_text = synonym_replacement(text, 2)

#     augmented_dict = {"Original text": orig_text,
#                       "After replacement with synonyms": synonymed_text}
#     return augmented_dict


def upload_image(contents):
    # Handle image upload logic if needed
    return {"message": "Image uploaded successfully"}

def preprocess_image(contents):
    # Convert bytes to image
    image = Image.open(io.BytesIO(contents))
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transformed_img = transform(image)
    
    # Convert the tensor back to a PIL image for display
    transformed_img = transforms.ToPILImage()(transformed_img)
    
    # Save the image to a BytesIO object
    img_byte_arr = io.BytesIO()
    transformed_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()  # Return the byte data of the image

def upload(data, file_type):
    if file_type == "text":
        return upload_text(data)  # Pass the contents directly
    elif file_type == "image":
        return upload_image(data)  # Call the updated upload_image function
    else:
        print(f"data type unknown: {file_type}") 

def preprocess(data, file_type):
    if file_type == "text":
        return preprocess_text(data)
    elif file_type == "image":
        return preprocess_image(data)  # Pass the image data directly
    else:
        print(f"data type unknown: {file_type}")
    
# def synonym_replacement(text, n=2):
#     words = text.split()
#     new_words = words[:]
#     random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
#     random.shuffle(random_word_list)
#     num_replaced = 0
#     for random_word in random_word_list:
#         synonyms = wordnet.synsets(random_word)
#         synonym = random.choice(synonyms).lemmas()[0].name()
#         new_words = [synonym if word == random_word else word for word in new_words]
#         num_replaced += 1
#         if num_replaced >= n:
#             break
#     return " ".join(new_words)

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
    if "Synonym Replacement" in options:
        print(f"before: {text}")
        text = synonym_replacement(text)
        print(f"after: {text}")

    if "Back Translation" in options:
        text = back_translation(text)
    if "Random Insertion" in options:
        text = random_insertion(text)
    if "Random Deletion" in options:
        text = random_deletion(text)
    return text

# def augment(data, file_type):
#     if file_type == "text":
#         data = " ".join(data.get("Tokenisation output",""))
#         return augment_text(data)
    # Add logic for image augmentation if needed
    