# OCR Skooldio

<img width="400" src="assets/skooldio-logo.png">

ยินดีต้อนรับสู่บทเรียน Optical Character Recognition (OCR) สอนโดย Skooldio! บทเรียนนี้ออกแบบมาเพื่อช่วยให้ผู้เริ่มต้นทดลองใช้เทคนิคการรู้จำอักขระด้วยแสง (OCR)
เหมาะสำหรับผู้เริ่มต้นทำ OCR หรือกำลังมองหาวิธีทำความเข้าใจว่าเครื่องมือและโมเดลต่างๆ ทำงานอย่างไร ที่เก็บข้อมูลนี้ให้บทนำที่ครอบคลุมเกี่ยวกับการตรวจจับและรู้จำข้อความ

OCR เป็นเทคโนโลยีที่สำคัญสำหรับการแปลงเอกสารประเภทต่างๆ เช่น เอกสารกระดาษที่สแกน ไฟล์ PDF หรือภาพที่ถ่ายด้วยกล้องดิจิทัล ให้เป็นข้อมูลที่สามารถแก้ไขและค้นหาได้
ที่เก็บข้อมูลนี้นำเสนอชุดเครื่องมือและทรัพยากรเพื่อช่วยให้คุณสำรวจและทำความเข้าใจเทคนิค OCR ต่างๆ

## เทคนิคที่ครอบคลุม

ในบทเรียน เราจะได้ทดลองเทคนิค OCR ที่หลากหลายโดยใช้เครื่องมือและโมเดลต่อไปนี้:

- Multimodal LLM (Gemini, ChatGPT): ทดลองใช้ Multimodal capability ของ Gemini และ ChatGPT สำหรับงาน OCR
- Google Vision API: เรียนรู้วิธีใช้ประโยชน์จากความสามารถ OCR จาก Google Cloud ผ่าน Vision API ซึ่งให้ความแม่นยำสูงและรองรับภาษาที่หลากหลาย
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): ทดลองใช้ไลบรารี OCR ใช้งานง่ายและมีน้ำหนักเบา รองรับหลายภาษา
- [Surya](https://github.com/VikParuchuri/surya): ไลบรารี่ OCR ที่รองรับหลายภาษาเช่นกัน
- DocVQA กับ Donut: Multimodal ที่สามารถตอบคำถามจากภาพเอกสาร (VQA) ด้วย Donut ซึ่งเป็นโมเดลที่ออกแบบมาโดยเฉพาะสำหรับการทำความเข้าใจและดึงข้อมูลจากเอกสาร
- Propritary OCR Model ของ Looloo Technology: ใช้โมเดลปิดของ Looloo Technology เพื่อทดสอบประสิทธิภาพของการทำ OCR

จริงๆแล้วยังมีไลบรารี่เกี่ยวกับ OCR อีกมากมาย สามารถทดลองดูเพิ่มเติมได้ที่ Github [zacharywhitley/awesome-ocr](https://github.com/zacharywhitley/awesome-ocr)

## ข้อมูลเพิ่มเติม

### ลงไลบรารี่ที่เกี่ยวข้อง

ถ้าทดสอบในเครื่องสามารถลงไลบรารี่ต่างๆได้ดังนี้

> \* สำหรับเครื่องที่เป็นระบบปฏิบัติการ windows **หรือ** เครื่องที่ไม่มีการ์ดจอ (GPU) ให้ลงรันคำสั่งด้านล่างก่อนเพื่อติดตั้ง `torch` ไลบรารี่ก่อน แล้วค่อยลงไลบรารี่อื่นๆ ตามปกติ

```sh
pip install torch torchvision torchaudio
```

สำหรับระบบปฏิบัติการอื่น เช่น macos หรือ linux สามารถลงไลบรารี่ที่เกี่ยวข้องด้วยคำสั่งด้านล่างได้เลย โดยไม่ต้องลง `torch` ก่อน

```sh
pip install -r requirements.txt
```

หรือถ้าเปิดผ่าน Colab สามารถลงไลบรารี่ที่เกี่ยวข้องผ่าน Notebook ได้เลย

### วิธีสร้าง API Key จาก OpenAI

- เปิดเว็บเบราว์เซอร์และไปที่ https://platform.openai.com/
- คลิกที่ปุ่ม "Sign up" ที่มุมขวาบนของหน้าเว็บ, ถ้ามีบัญชีอยู่แล้ว ให้คลิก "Log in" แทน
- หลังจากเข้าสู่ระบบแล้ว ให้คลิกที่ชื่อผู้ใช้ของคุณที่มุมขวาบน และเลือก "Your Profile" ตามด้วย "User API keys"
- ในหน้า API keys ให้คลิกปุ่ม "Create new secret key" แนะนำตั้งชื่อให้กับ API key ของคุณ การตั้งชื่อไม่บังคับ แต่แนะนำให้ตั้งเพื่อจำได้ง่าย จากนั้นระบบจะแสดง API key ของคุณ ให้คัดลอกและเก็บไว้ในที่ปลอดภัย (สำคัญ: คุณจะเห็น API key นี้ครั้งเดียวเท่านั้น ถ้าคุณทำหาย จะต้องสร้างใหม่)
- เมื่อคัดลอก API key เรียบร้อยแล้ว ให้คลิก "Done" จากนั้นจะสามารถใช้ API key นี้ในการเข้าถึงบริการต่างๆ ของ OpenAI ผ่าน API ได้แล้ว

<img width="700" src="assets/chatgpt.png">

หลังจากได้ API key แล้วเราสามารถเรียกใช้ ChatGPT ผ่าน Python ได้ด้วยคำสั่ง

```py
from openai import OpenAI
from IPython.display import display

OPENAI_API_KEY = "<api-key>" # ใส่ API key ที่นี่
client = OpenAI(api_key=OPENAI_API_KEY)

def get_completion(prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

get_completion("Why is sky blue?")
```

### วิธีสร้าง JSON Credential จาก Google Cloud Platform

- ไปที่ https://cloud.google.com/vertex-ai และเข้าไปยัง "Try it in console"
- จากนั้น Enable Vertex AI service เพื่อให้สามารถใช้งานโมเดล Gemini ได้ ถ้าสนใจเล่นก่อนให้กดไปในแถบ "Freeform" เพื่อทดลองเล่นก่อนได้
- จากนั้น Enable Cloud Vision API ที่ https://cloud.google.com/vision?hl=en เพื่อเปิดใช้งาน OCR

<img width="700" src="assets/vertex_ai.png">

- จากนั้นกลับไปที่ https://console.cloud.google.com/ สร้าง Project ใหม่ถ้ายังไม่เคยสร้าง
- เข้าไปยัง APIs & Services และกดปุ่ม "+ Create Credentials > Service Account" เพื่อสร้าง Credential ใหม่ ในข้อ "Grant this service account access" ให้เลือก Vertex AI Administrator และเพิ่ม Cloud Vision API
- เมื่อสร้างเสร็จเรียบร้อย กลับมาที่หน้า Service Account และไปที่หัวข้อ Keys จากนั้นกด Add Key > Create new key เพื่อสร้าง JSON ไฟล์
- จากนั้นไฟล์จะถูกดาวน์โหลดเข้ามาในเครื่อง และเราสามารถใช้ Python ในการเรียกใช้ Gemini ได้ดังนี้

```py
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

project_name = "<project-name>"
credentials = service_account.Credentials.from_service_account_file("<path-to-json>")
aiplatform.init(project=project_name, credentials=credentials)

model = GenerativeModel("gemini-1.5-flash-001")
print(model.generate_content("Why is sky blue?"))
```

## วิธีเรียกใช้โมเดลจาก Ollama ด้วย Langchain

- ดาวน์โหลด [Ollama](https://ollama.com/) และจากนั้นโหลดโมเดล llama 3.1 และ llama 3.2 ด้วยคำสั่ง

```sh
ollama run llama3.1
ollama run llama3.2
```

- จากนั้นลงไลบรารี่ที่เกี่ยวข้อง

```sh
!pip install langchain
!pip install langchain_community
```

- และทดลองรันโมเดลด้วย Python โดยการใช้ฟังก์ชั่นตามด้านล่าง เราจะใช้โมเดลเปิดเหล่านี้ในการจัดรูปของ output ที่ได้จาก OCR เพื่อนำไปใช้งานต่อได้

```py
# Code from https://stackoverflow.com/a/78430197/3626961
from langchain_community.llms import Ollama
from langchain import PromptTemplate # Added

llm = Ollama(model="llama3.1", stop=["<|eot_id|>"]) # Added stop token

def get_model_response(user_prompt, system_prompt):
    # NOTE: No f string and no whitespace in curly braces
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Added prompt template
    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )

    # Modified invoking the model
    response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))

    return response

# Example
user_prompt = "What is 1 + 1?"
system_prompt = "You are a helpful assistant doing as the given prompt."
get_model_response(user_prompt, system_prompt)
```

## วิธีสร้างรัน streamlit สำหรับการทดสอบโมเดล

- รันไฟล์ `notebooks/05_streamlit_visualiztion.py` ด้วยคำสั่ง

```sh
streamlit run notebooks/05_streamlit_visualization.py
```

- จากนั้นโปรแกรมจะเปิดหน้าเว็บให้เราโดยอัตโนมัติเพื่อให้เราทดสอบโมเดลได้

<img width="1000" src="assets/streamlit_idle.png">

- อัพโหลดไฟล์รูปของเอกสารที่ต้องการทดสอบ

<img width="1000" src="assets/streamlit_uploaded.png">

- กดปุ่ม "Run" เพื่อทำการ detect ข้อความในรูป และ รันโมเดล OCR เพื่ออ่านข้อความในรูป

\* หมายเหตุ: หากรูปที่อัพโหลดเป็นรูปของข้อความที่ถูกตัดออกมาจากเอกสารแล้ว สามารถติ๊กเลือก "Recognition Only" เพื่อทำ OCR ได้เลย

<img width="700" src="assets/streamlit_predicted.png">
