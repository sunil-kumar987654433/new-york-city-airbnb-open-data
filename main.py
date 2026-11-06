import joblib
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from decimal import Decimal
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = ['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
       'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365']
loaded_model = joblib.load(filename='random_forest_room_classifier.pkl')

class NeighbourhoodGroup(str, Enum):
    Manhattan = 'Manhattan'
    Brooklyn = 'Brooklyn'
    Queens = 'Queens'
    Bronx = 'Bronx'
    Staten_Island = 'Staten Island'


class HomeFeature(BaseModel):
    neighbourhood_group: str = Field(description="Name of area")
    neighbourhood: str = Field(description="neighbourhood area.")
    latitude: Decimal = Field(max_digits=30, decimal_places=15)
    longitude: Decimal = Field(max_digits=30, decimal_places=15)
    price: Decimal = Field(max_digits=10, decimal_places=2)
    minimum_nights: int = Field(gt=0)
    number_of_reviews: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    calculated_host_listings_count: float = Field(ge=0)
    availability_365: int = Field(gt=0)

from pprint import pprint
templates = Jinja2Templates(directory='templates')


@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = dict(await request.form())

    try:
        feature = HomeFeature(**form)
    except ValidationError as e:
        print(e)
    

    row = pd.DataFrame([feature.model_dump()])

    predictions = loaded_model.predict(row)
    probability  = loaded_model.predict_proba(row)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context = {
            "form": form,
            "room_type": str(predictions[0]),
            "probability": dict(zip(loaded_model.classes_.tolist(), probability[0].round(2).tolist()))
        }
    )
    