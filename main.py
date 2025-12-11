from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import create_mobile_net_v3
import torch
from PIL import Image
import io
import numpy as np
import time

app=FastAPI()

# enable CORS for our frontend 
# this is used for communication between frontend and backend

# specifying origins allowed to communicate with backend if needed can be specified in list
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True, # allow cookies
    allow_methods=['*'], # allow all HTTP methods(GET,Post, etc) for cross origin requests
    allow_headers=['*'], # allows all headers
)

# class names 
class_names=['aloo gobi', 
'aloo methi',
'aloo mutter',
'aloo paratha',
'amritsari kulcha',
'anda curry',
'balushahi',
'banana chips',
'besan laddu',
'bhindi masala',
'biryani',
'boondi laddu',
'chaas',
'chana masala',
'chapati',
'chicken pizza',
'chicken wings',
'chikki',
'chivda',
'chole bhature',
'dabeli',
'dal khichdi',
'dhokla',
'falooda',
'fish curry',
'gajar ka halwa',
'garlic bread',
'garlic naan',
'ghevar',
'grilled sandwich',
'gujhia',
'gulab jamun',
'hara bhara kabab',
'idiyappam',
'idli',
'jalebi',
'kaju katli',
'khakhra',
'kheer',
'kulfi',
'margherita pizza',
'masala dosa',
'masala papad',
'medu vada',
'misal pav',
'modak',
'moong dal halwa',
'murukku',
'mysore pak',
'navratan korma',
'neer dosa',
'onion pakoda',
'palak paneer',
'paneer masala',
'paneer pizza',
'pani puri',
'paniyaram',
'papdi chaat',
'patrode',
'pav bhaji',
'pepperoni pizza',
'phirni',
'poha',
'pongal',
'puri bhaji',
'rajma chawal',
'rasgulla',
'rava dosa',
'sabudana khichdi',
'sabudana vada',
'samosa',
'seekh kebab',
'set dosa',
'sev puri',
'solkadhi',
'steamed momo',
'thali',
'thukpa',
'uttapam',
'vada pav']

# create instance of model and transforms
model,transforms=create_mobile_net_v3(class_names)

# load the model weights
try:
    model.load_state_dict(torch.load('MobileNetV3_100_data_15_epochs_test_acc88_with_unfreezing_3_layers.pth', map_location=torch.device('cpu')))
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("⚠️ Warning: Model file not found. Running with untrained weights.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    try:
        #start time
        start_time=time.time()
        # set model to eval mode
        model.eval()
        
        with torch.inference_mode():
            
            # read and preprocess image
            contents=await file.read()
            image=Image.open(io.BytesIO(contents)).convert('RGB')
            
            # preprocess image
            processsed_image=transforms(image).unsqueeze(0)
            
            # make prediction
            predictions=model(processsed_image)
            
            # get top 5 predictions
            probs=torch.nn.functional.softmax(predictions, dim=1)
            top5_probs, top5_indices=torch.topk(probs, k=5)
            top5_probs=top5_probs.squeeze().tolist()
            top5_indices=top5_indices.squeeze().tolist()
            top5_class_names=[class_names[idx] for idx in top5_indices] 
            # end time
            end_time=time.time()    
            
            # return predictions and processing time
            return {
                "predictions": list(zip(top5_class_names, top5_probs)),
                "processing_time": end_time - start_time
            }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"error": str(e)}
        
@app.get('/')
async def root():
    return {'message':'Indian Food Vision API is running!'}
