from fastapi import *
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle


with open("model.pkl","rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl","rb") as f:
    encoder = pickle.load(f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request:Request):
    return templates.TemplateResponse("index.html",{'request':request})

@app.post("/submit")
async def predict(request:Request,Anual_salary : int = Form(...),
                  Loan_Amount : float = Form(...),
                  Creadit_Score : int = Form(...)):
    input_data = np.array([[Anual_salary,Loan_Amount,Creadit_Score]])
    Prediction = model.predict(input_data)
    decoded = encoder.inverse_transform(Prediction)[0]
    return templates.TemplateResponse("index.html",{'request':request,
                                                    'Prediction':decoded}
                                                    )

