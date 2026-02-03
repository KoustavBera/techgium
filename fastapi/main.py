from typing import Union, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.engine_cardiovascular import CardioRiskEngine # type: ignore
from models.engine_derma import DermatologyRiskEngine # type: ignore
from models.engine_neurofunctionalrisk import NeuroFunctionalRiskEngine # type: ignore
from models.engine_posturerisk import PostureRiskEngine # type: ignore
from models.engine_respiratory import RespiratoryRiskEngine # type: ignore
from models.engine_ris_resonator import RISResonatorEngine # type: ignore
from models.aggregator_hri import HealthRiskIndex # type: ignore
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from ml.fusion.fusion_inference import FusionEngine
from routes.health_assessment import router as health_router

app = FastAPI(
    title="Health Chamber Walkthrough Diagnosis API",
    description="High-accuracy IoT sensor-based health risk assessment",
    version="2.0.0"
)

# Register health assessment routes
app.include_router(health_router)

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

class RISResonatorAssessment(BaseModel):
    rf_backscatter_data: List[List[float]]  # 2D array as list of lists
    ris_phase_config: List[float]           # RIS element phases
    target_distance: float                  # Distance to subject (meters)
    frequency_range: List[float] = [2.4e9, 2.5e9]  # RF frequency range
    signal_quality: float = 1.0
    calibration_data: Optional[dict] = None
    

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

@app.post("/assess/ris-resonator")
def assess_ris_resonator(data: RISResonatorAssessment):
    """
    RIS-integrated passive resonator assessment endpoint
    Processes RF backscatter data for non-contact vital sign monitoring
    """
    try:
        import numpy as np
        
        # Convert lists to numpy arrays
        rf_data = np.array(data.rf_backscatter_data)
        ris_config = np.array(data.ris_phase_config)
        
        engine = RISResonatorEngine(
            rf_backscatter_data=rf_data,
            ris_phase_config=ris_config,
            target_distance=data.target_distance,
            frequency_range=tuple(data.frequency_range),
            signal_quality=data.signal_quality,
            calibration_data=data.calibration_data
        )
        
        result = engine.run()
        
        # Add RIS-specific metadata
        result["ris_metadata"] = {
            "num_elements": len(ris_config),
            "frequency_range_ghz": [f/1e9 for f in data.frequency_range],
            "target_distance_m": data.target_distance,
            "data_shape": rf_data.shape
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e), "system": "RIS_Passive_Resonator"}
# To run the server:
# uvicorn main:app --reload

fusion_engine = FusionEngine()


class FusionInput(BaseModel):
    embeddings: List[List[float]]   # [[cardio_emb], [resp_emb]]


@app.post("/assess/fusion")
def assess_fusion(data: FusionInput):
    return fusion_engine.run(data.embeddings)
    
    
@app.post("/assess/complete")
def complete_assessment(data: dict):
    try:
        embeddings = [
            data["cardio"]["embedding"],
            data["respiratory"]["embedding"]
        ]

        hri = HealthRiskIndex()
        return hri.run(embeddings)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        