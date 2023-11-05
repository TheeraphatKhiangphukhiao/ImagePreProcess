from typing import Union
from fastapi import FastAPI, Request 
from pydantic import BaseModel
import cv2
from matplotlib import pyplot as plt
import base64
import numpy as np
import base64
import requests

app = FastAPI()
 
@app.get("/")
def read_root():
    return {"Hello": "Hello World !!!"}

# API ในส่วนของ Container 1 โดยจะรันที่ port 8080 เป็นตัวรับรูปภาพ Input ที่เป็น base64 จาก front end
# โดยจะเป็นตัวทำ pre-process image data เพื่อจะนำข้อมูลที่ได้ส่งไปยัง Container 2
@app.post("/api/image/pre-process")
async def Image_PreProcess(image_base64 : Request):

    width = 28 # ทำการปรับขนาดความกว้างความสูงของรูปภาพทั้งหมดให้เป็น 28 x 28 Pixel เพื่อให้ตรงกับข้อมูลรูปภาพที่ทำการ Train Model
    image_base64_json = await image_base64.json() # ทำการเเปลงข้อมูลที่ส่งมาให้เป็น json
    
    # ทำการเเปลงข้อมูลรูปภาพ ให้สามารถใช้ในการ resize 28 x 28 Pixel เพื่อให้ตรงกับข้อมูลรูปภาพที่ทำการ Train Model
    image_base64_json['image_base64'] = image_base64_json['image_base64'].split(',')[1]
    image_data = base64.b64decode(image_base64_json['image_base64']) # ถอดรหัส Base64 เพื่อให้เราได้ข้อมูลรูปภาพในรูปแบบ binary data
    image_array = np.frombuffer(image_data, np.uint8) # แปลงข้อมูลรูปภาพในรูปแบบ binary data เป็น NumPyarray
    img = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB) # แปลง NumPyarray ที่เก็บรูปภาพในรูปแบบ binary data ให้เป็นรูปภาพเเบบ RGB 
    img = cv2.resize(img ,(width, width)) # resize 28 x 28 Pixel
    img = img.astype('float32') / 255.0 # ทําการ normalize pixel จากเดิมค่าอยู่ระหว่าง 0-255 ให้อยู่ในช่วง 0-1 เพื่อให้การประมวลผลนั้น ง่ายและเร็ว
    #print(img)
    #plt.imshow(img, cmap = 'gray')
    #plt.show()
    
    # เมื่อทำการ resize เสร็จก็จะเเปลงรูปภาพให้เป็น base64 อีกครั้งเพื่อจะส่งไปให้กับ Container 2 ทำการ predict ต่อไป
    v, buffer = cv2.imencode(".jpg", img) 
    img_str = base64.b64encode(buffer).decode("utf-8") # เเปลงรูปภาพเป็น base64
    
    json_data = {"image_base64" : img_str}
    result = requests.post('http://172.17.0.2:80/api/image/predicted', json=json_data) # ทำการส่งข้อมูลรูปภาพไปให้กับ Container 2
    myresult = result.json()['Flower Type']
    return {"Flower Type": myresult}