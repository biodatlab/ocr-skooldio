# Projects

## Gradio Application

ปรับ `MODEL_PATH` และ `LOGO_PATH` ไปยัง path ที่ถูกต้อง จากนั้นรัน Gradio application ด้วยคำสั่ง:

``` sh
gradio webapp.py
```

หลังจากทีรันแล้วไปที่ `http://localhost:7860/` เพื่อใช้ web application

## API

หรืออาจทำเป็น API endpoint โดยใช้ไลบรารี่ `FastAPI` เพื่อสร้าง API ที่ port 8000 และส่งภาพเพื่ออ่านก็ได้

``` sh
python api.py
```

```sh
curl -X POST "http://localhost:8000/detect/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

## Evaluation Notebook

สามารถดูโค้ดสรุปการใช้ detection model เพื่ออ่านกล่องข้อความและส่งให้ recognizer ได้ใน `end_to_end.ipynb`
