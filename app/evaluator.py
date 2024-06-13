import os
import platform
import re
import shutil
import subprocess
from sys import exit
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from io import BytesIO
from cachetools.func import lru_cache
from collections import namedtuple
import functools
import json
import six


from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


Serialized = namedtuple('Serialized', 'json')
client_key_path= r"app/key.json"
current_os = platform.system().lower()
print(f"Current OS: {current_os}")

if current_os == 'windows':
    
    poppler_executable = 'pdfinfo.exe'   
    poppler_installation_path = os.path.join(os.environ['ProgramFiles(x86)'], 'poppler-23.11.0', 'Library', 'bin')
    

    if not os.path.exists(poppler_installation_path):
        print("Poppler not found. Attempting to install Poppler on Windows...")

        # Download Poppler zip file (adjust the URL accordingly)
        # Software library rendering thre pdf. extract text image and other pdf document 
        poppler_zip_url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip"
        subprocess.run(['curl', '-L', '-o', 'poppler.zip', poppler_zip_url])

        # Extract Poppler zip file to a directory (adjust the path accordingly)
        subprocess.run(['tar', '-xf', 'poppler.zip', '-C', os.environ['ProgramFiles(x86)']])

        # Add Poppler installation directory to PATH (modify the path accordingly)
        os.environ['PATH'] += os.pathsep + poppler_installation_path
    elif not shutil.which(poppler_executable):
        print("You already have Poppler installed. Please add the Poppler installation directory to PATH.")
        print(f"Example: set PATH=%PATH%;{poppler_installation_path}")
        

def hashable_cache(cache):
    def hashable_cache_internal(func):
        def deserialize(value):
            if isinstance(value, Serialized):
                return json.loads(value.json)
            else:
                return value

        def func_with_serialized_params(*args, **kwargs):
            _args = tuple([deserialize(arg) for arg in args])
            _kwargs = {k: deserialize(v) for k, v in six.viewitems(kwargs)}
            return func(*_args, **_kwargs)

        cached_func = cache(func_with_serialized_params)

        @functools.wraps(func)
        def hashable_cached_func(*args, **kwargs):
            _args = tuple([
                Serialized(json.dumps(arg, sort_keys=True))
                if type(arg) in (list, dict) else arg
                for arg in args
            ])
            _kwargs = {
                k: Serialized(json.dumps(v, sort_keys=True))
                if type(v) in (list, dict) else v
                for k, v in kwargs.items()
            }
            return cached_func(*_args, **_kwargs)
        hashable_cached_func.cache_info = cached_func.cache_info
        hashable_cached_func.cache_clear = cached_func.cache_clear
        return hashable_cached_func

    return hashable_cache_internal

# Function to convert PDF to images and extract text
@functools.lru_cache(maxsize=None)
def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    return [extract_text_from_image(image) for image in images]

def extract_text_from_image(image_content):
    # Set up the Google Cloud Vision API client with the service account key
    
    client = vision_v1.ImageAnnotatorClient.from_service_account_file(client_key_path)

    # Convert PIL(Python Image Library) Image to bytes 
    with BytesIO() as byte_io:
        image_content.save(byte_io, format='PNG')
        byte_value = byte_io.getvalue()

    # Create Image message with bytes content
    input_image = types.Image(content=byte_value)
    response = client.text_detection(image=input_image)


    texts = response.full_text_annotation
    return texts.text.replace('\n', ' ')

@hashable_cache(lru_cache())
def separate_answers(texts):
    separated_answers = {}

    last_answer = None
    for i, text in enumerate(texts):
        answers = re.split(r'(?=\s*Answer \d+\s*:)', text)

        for answer in answers:
            match = re.match(r'\s*Answer \d+\s*:', answer)
            if match:
                answer_number = match.group(0).replace(' :', ':')
                text = answer[len(answer_number):].strip()
                last_answer = f"{i + 1}_{answer_number}"
                separated_answers[last_answer] = text
            elif last_answer is not None:
                separated_answers[last_answer] += f" {answer.strip()}"
    
    return separated_answers

@hashable_cache(lru_cache())
def calculate_similarity(teacher_answers, student_answers):
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = {}

    for student_key, student_answer in student_answers.items():
        matching_teacher_key = next((key for key in teacher_answers if student_key.split('_')[1] in key), None)
        teacher_answer = teacher_answers.get(matching_teacher_key, "")

        student_tokens = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
        teacher_tokens = tokenizer(teacher_answer, return_tensors="pt", padding=True, truncation=True)
        student_embedding = model(**student_tokens).last_hidden_state.mean(dim=1).detach().numpy()
        teacher_embedding = model(**teacher_tokens).last_hidden_state.mean(dim=1).detach().numpy()

        similarity_score = cosine_similarity(student_embedding, teacher_embedding)[0][0]
        scores[student_key] = similarity_score

    return scores