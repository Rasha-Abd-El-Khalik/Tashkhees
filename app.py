import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import re, torch
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from langchain_google_genai import ChatGoogleGenerativeAI

# === MARBERT ===
tokenizer = AutoTokenizer.from_pretrained("./marbert_finetuned_lora")
base_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT", num_labels=5)
model = PeftModel.from_pretrained(base_model, "./marbert_finetuned_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key="AIzaSyCYJyangnXI9r_MiOf-6GTXgyz804Oi2II",
)


def camel_clean(text):
    text = dediac_ar(str(text))
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    en2ar = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    text = text.translate(en2ar)
    text = re.sub(r'[^\u0600-\u06FF\s.,;:!?()\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ar-EG")
        except:
            text = "⚠️ تعذر التعرف على الصوت، يرجى المحاولة مجددًا"
    return text

def predict_text(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        max_prob, pred = torch.max(probs, dim=-1)
    return pred.item(), max_prob.item()


def map_class(pred):
    mapping = {
        0: "أمراض الجهاز التنفسي",
        1: "الغدد الصماء",
        2: "جراحة العظام",
        3: "جراحة عامة",
        4: "مرض السكري"
    }
    return mapping.get(pred, "خارج الفئات")


def generate_category_with_gemini(question):
    prompt = f"""
استخرج فقط التخصص الطبي المناسب للسؤال التالي بكلمة أو كلمتين كحد أقصى.
مثال:
- "أمراض القلب"
- "طب الأطفال"
- "جراحة عامة"
- "العناية بالشعر"
- "السكري"

السؤال: {question}
اكتب الفئة فقط بدون أي شرح إضافي.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def generate_answer_with_gemini(question, max_tokens=100):
    prompt = f"""
أنت طبيب متخصص ذو خبرة عالية. أجب على أي سؤال طبي بطريقة احترافية، علمية، دقيقة وواضحة:
- استخدم العربية الفصحى فقط. **لا تجب أبدًا بالإنجليزية.**
- لا تتجاوز الإجابة 250 حرف.
- ركّز على التشخيص المبدئي أو التوصية العملية المباشرة.
- أجب مباشرة دون إعادة السؤال أو إضافة تفاصيل غير ضرورية.
- لا تستخدم اختصارات أو مصطلحات غامضة.

السؤال: {question}
أجب الآن مباشرة واحترافيًا باللغة العربية فقط، **لا تُجب بالإنجليزية**:
"""
    response = llm.invoke(prompt)
    return response.content

# === تحويل النص إلى صوت ===
def text_to_speech(text, filename="output.mp3"):
    if text.strip() == "":
        return None
    tts = gTTS(text=text, lang="ar")
    tts.save(filename)
    return filename

THRESHOLD = 0.90
def process(audio_file):
    if audio_file is None:
        return "—", None, "🚫 لم يتم إدخال صوت", None

    clean = camel_clean(speech_to_text(audio_file))

    # MARBERT prediction
    pred, confidence = predict_text(clean)
    mapped = map_class(pred)

    if confidence >= THRESHOLD:
       
        category_text = mapped
        answer_text = generate_answer_with_gemini(clean)
    else:
  
        category_text = generate_category_with_gemini(clean)
        answer_text = generate_answer_with_gemini(clean)


    category_speech = text_to_speech(category_text, filename="category.mp3")
    answer_speech = text_to_speech(answer_text, filename="answer.mp3")

    return category_text, category_speech, answer_text, answer_speech

with gr.Blocks(css="""
    body, .gradio-container {min-height: 100vh; background: #f7f9fb; font-family: 'Cairo', sans-serif;}
    .header-bar, .footer-bar {width:100%; text-align:center; padding:15px; border-radius:12px;
                               background:linear-gradient(90deg, #00796b, #00acc1); color:white;
                               box-shadow:0 2px 10px rgba(0,0,0,0.15);}
    .footer-bar {margin-top:30px; font-size:0.85em; border-radius:0; padding:8px; box-shadow:none;}
    .header-text {font-size:1.8em; font-weight:bold; margin:0;}
    .sub-text {font-size:1em; color:#e0f2f1; margin-top:5px;}
    .card {border-radius:12px; padding:15px; background:#ffffff; color:#2c3e50;
           box-shadow:0px 4px 12px rgba(0,0,0,0.1);}
    .gr-textbox label, .gr-audio label {color:#00796b !important; font-weight:bold;}
""") as demo:

    # Header
    gr.HTML("""
    <div class="header-bar">
      <h1 class="header-text">🩺 تشخيص | Tashkhees</h1>
      <p class="sub-text">واجهة ذكية لتحويل كلامك الطبي إلى فئة وإجابة مباشرة</p>
    </div>
    """)

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ سجل صوتك هنا")

    with gr.Row():
        with gr.Column():
            category_output = gr.Textbox(label="🏷️ الفئة المصنفة", interactive=False, elem_classes=["card"])
            category_audio = gr.Audio(label="🔊 الفئة صوت", type="filepath")
        with gr.Column():
            answer_output = gr.Textbox(label="💡 الإجابة", interactive=False, lines=5, elem_classes=["card"])
            answer_audio = gr.Audio(label="🔊 الإجابة صوت", type="filepath")

    # Footer
    gr.HTML("""
    <div class="footer-bar">
      <p>© 2025 تشخيص | Tashkhees Project | Designed with ❤️</p>
    </div>
    """)

    # Link function
    audio_input.change(fn=process, inputs=audio_input,
                       outputs=[category_output, category_audio, answer_output, answer_audio])

demo.launch(share=True)
