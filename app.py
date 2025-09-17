import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import re, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("./marbert_finetuned_lora")
base_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT", num_labels=5)
model = PeftModel.from_pretrained(base_model, "./marbert_finetuned_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text cleaning
def camel_clean(text):
    en2ar = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    text = text.translate(en2ar)
    text = re.sub(r'[^\u0600-\u06FF\s.,;:!?()\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Speech-to-text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ar-EG")
        except:
            text = "⚠️ تعذر التعرف على الصوت، يرجى المحاولة مجددًا"
    return text

# Prediction
def predict_text(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred

# Map class and colors
def map_class(pred):
    mapping = {
        0: ("أمراض الجهاز التنفسي", "#FFCDD2"),
        1: ("الغدد الصماء", "#C8E6C9"),
        2: ("جراحة العظام", "#BBDEFB"),
        3: ("جراحة عامة", "#FFE0B2"),
        4: ("مرض السكري", "#D1C4E9")
    }
    return mapping.get(pred, ("غير معروف", "#CFD8DC"))

# Text-to-speech
def text_to_speech(text):
    if text.strip() == "":
        return None
    tts = gTTS(text=text, lang="ar")
    tts.save("output.mp3")
    return "output.mp3"

# Full processing
def process(audio_file):
    if audio_file is None:  # لو المستخدم لغى التسجيل
        return "🚫 لم يتم إدخال صوت", "—", None, "#9e9e9e"  # نص + فئة فاضية + رمادي فاتح
    
    text = speech_to_text(audio_file)
    clean = camel_clean(text)
    pred = predict_text(clean)
    mapped, color = map_class(pred)
    speech_file = text_to_speech(mapped)
    return text, mapped, speech_file, color


# Create UI
with gr.Blocks(css="""
    body, .gradio-container {
        min-height: 100vh;
        background: #f7f9fb;
        font-family: 'Cairo', sans-serif;
    }
    .header-bar, .footer-bar {
        width: 100%;
        text-align: center;
        padding: 15px;
        border-radius: 12px;
        background: linear-gradient(90deg, #00796b, #00acc1);
        color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .footer-bar {
        margin-top: 30px;
        font-size: 0.85em;
        border-radius: 0;
        padding: 8px;
        box-shadow: none;
    }
    .header-text {
        font-size: 1.8em;
        font-weight: bold;
        margin: 0;
    }
    .sub-text {
        font-size: 1em;
        color: #e0f2f1;
        margin-top: 5px;
    }
    .card {
        border-radius: 12px;
        padding: 15px;
        background: #ffffff;
        color: #2c3e50;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .gr-textbox label, .gr-audio label {
        color: #00796b !important;
        font-weight: bold;
    }
""") as demo:

    # Header
    gr.HTML("""
    <div class="header-bar">
      <h1 class="header-text">🩺 تشخيص | Tashkhees</h1>
      <p class="sub-text">واجهة ذكية لتحويل كلامك الطبي إلى تشخيص مبدئي دقيق</p>
    </div>
    """)

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ سجل صوتك هنا")

    with gr.Row():
        text_output = gr.Textbox(label="📝 النص الأصلي", interactive=False, elem_classes=["card"])
        mapped_output = gr.Textbox(label="🏷️ الفئة المصنفة", interactive=False, elem_classes=["card"])
        audio_output = gr.Audio(label="🔊 النص محوّل إلى صوت")

    class_color = gr.Textbox(visible=False)

    # Footer
    gr.HTML("""
    <div class="footer-bar">
      <p>© 2025 تشخيص | Tashkhees Project | Designed with ❤️</p>
    </div>
    """)

    # Link function
    audio_input.change(fn=process, inputs=audio_input,
                       outputs=[text_output, mapped_output, audio_output, class_color])

demo.launch(share=True)