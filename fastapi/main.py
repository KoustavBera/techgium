from typing import Union, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.engine_cardiovascular import CardioRiskEngine # type: ignore
from models.engine_derma import DermatologyRiskEngine # type: ignore
from models.engine_neurofunctionalrisk import NeuroFunctionalRiskEngine # type: ignore
from models.engine_posturerisk import PostureRiskEngine # type: ignore
from models.engine_respiratory import RespiratoryRiskEngine # type: ignore
from models.aggregator_hri import HealthRiskIndex # type: ignore
app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

# Request models
class CardioAssessment(BaseModel):
	age: int
	heart_rate: float
	spo2: float
	ecg_rr_intervals: Optional[list[float]] = None
	signal_quality: float = 1.0
	
class RespiratoryAssessment(BaseModel):
	age: int
	respiratory_rate: float
	spo2: float
	nasal_airflow_variability: Optional[float] = None
	cough_present: bool = False
	signal_quality: float = 1.0

class DermatologyAssessment(BaseModel):
    lesion_detected: bool
    lesion_confidence: float
    lesion_area_ratio: float
    lesion_count: int
    image_quality: float = 1.0

class NeuroFunctionalAssessment(BaseModel):
	blink_rate_deviation: float
	blink_asymmetry: float
	facial_asymmetry: float
	head_tremor: float
	gaze_instability: float
	signal_quality: float = 1.0
	
class PostureAssessment(BaseModel):
	shouder_asymmetry: float
	hip_asymmetry: float
	spine_deviation: float
	head_tilt: float
	gait_instability: Optional[float] = None
	signal_quality: float = 1.0
    

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id, "price": item.price}

    
# Main Engine Endpoints (formal input - formal output)
@app.post("/assess/cardiovascular")
def assess_cardiovascular(data: CardioAssessment):
	try:
		engine = CardioRiskEngine(**data.dict())
		return engine.run()
	except Exception as e:
		return {"error": str(e)}


@app.post("/asses/respiratory")
def assess_respiratory(data: RespiratoryAssessment):
	try:
		engine = RespiratoryRiskEngine(**data.dict())
		return engine.run()
		
	except Exception as e:
		return {"error": str(e)}
		
@app.post("/asses/dermatology")
def assess_dermatology(data: DermatologyAssessment):
	try:
		engine = DermatologyRiskEngine(**data.dict())
		return engine.run()
		
	except Exception as e:
		return {"error": str(e)}
	
@app.post("/asses/neurofunctional")
def assess_neurofunctional(data: NeuroFunctionalAssessment):
	try:
		engine = NeuroFunctionalRiskEngine(**data.dict())
		return engine.run()
		
	except Exception as e:
		return {"error": str(e)}

@app.post("/asses/posture")
def assess_posture(data: PostureAssessment):
	try:
		engine = PostureRiskEngine(**data.dict())
		return engine.run()
		
	except Exception as e:
		return {"error": str(e)}
		
@app.post("/assess/complete")
def complete_health_assessment(assessments: List[dict]):
    try:
        hri = HealthRiskIndex(assessments)
        return hri.run()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# To run the server:
# uvicorn main:app --reload
