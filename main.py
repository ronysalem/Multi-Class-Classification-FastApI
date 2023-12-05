from fastapi import FastAPI,Request
import uvicorn
import numpy as np 
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
from os import getenv

app = FastAPI()

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("Ronysalem/Bert-Multiclass-Classification")
    return tokenizer,model

tokenizer,model =get_model()

id2label= {0: 'Computer Science',
 1: 'Physics',
 2: 'Mathematics',
 3: 'Statistics',
 4: 'Quantitative Biology',
 5: 'Quantitative Finance'}

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text' in data:
        user_input = data['text']
        
        encoding = tokenizer(user_input, return_tensors="pt")
        encoding = {k: v.to(model.device) for k, v in encoding.items()}
        
        # Forward pass through the model
        outputs = model(**encoding)
        
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        
        # turn predicted id's into actual label names
        predicted_labels = [id2label[id] for id, label in enumerate(predictions) if label == 1.0]
        
        response = {'Reccived Text': user_input , 'Prediction': predicted_labels }
        
    else:
        response = {'Reccived Text': 'No Text Found' }
        
    return response

if __name__ == "__main__":
    port = int(getenv("PORT",8000))
    uvicorn.run("main:app", host = '0.0.0.0', port = port, reload=True)
        
    
    


