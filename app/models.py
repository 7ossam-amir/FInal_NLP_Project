import threading
import queue
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# Global variables
detection_model = None
blip_processor = None
blip_model = None
translation_tokenizer = None
translation_model = None
model_queue = queue.Queue()
models_loaded = False

def load_models():
    global detection_model, blip_processor, blip_model, translation_tokenizer, translation_model, models_loaded
    print("Loading detection model...")
    detection_model = YOLO('yolov8l.pt')
    detection_model.to('cuda')
    print("Loading image captioning model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("Loading translation model...")
    translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    print("All models loaded!")
    models_loaded = True
    model_queue.put(True)

# Start model loading in a background thread
threading.Thread(target=load_models, daemon=True).start()

def get_models():
    return {
        'detection_model': detection_model,
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'translation_tokenizer': translation_tokenizer,
        'translation_model': translation_model,
        'models_loaded': models_loaded,
        'model_queue': model_queue
    }