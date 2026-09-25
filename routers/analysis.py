from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import ml_model
from database import get_db, log_request

router = APIRouter()


@router.post("/analyze/{coin_id}")
def analyze_ticker(coin_id: str, db: Session = Depends(get_db)):
    coin_id = coin_id.strip().lower()  # coingecko ids are lowercase, e.g. "bitcoin"
    try:
        result = ml_model.train_and_save(coin_id)
    except ValueError as e:
        log_request(db, coin_id, "/analyze", "failed", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # catch-all so a coingecko hiccup doesn't 500 without a trace in the log
        log_request(db, coin_id, "/analyze", "failed", str(e))
        raise HTTPException(status_code=502, detail=f"upstream data fetch failed: {e}")

    log_request(db, coin_id, "/analyze", "success", f"trained on {result['rows_trained']} rows")
    return result


@router.get("/anomalies/{coin_id}")
def get_anomalies(coin_id: str, db: Session = Depends(get_db)):
    coin_id = coin_id.strip().lower()
    try:
        data = ml_model.predict_anomalies(coin_id)
    except FileNotFoundError as e:
        log_request(db, coin_id, "/anomalies", "failed", str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_request(db, coin_id, "/anomalies", "failed", str(e))
        raise HTTPException(status_code=500, detail=f"could not compute anomalies: {e}")

    anomalies = data["anomalies"]
    log_request(db, coin_id, "/anomalies", "success", f"{len(anomalies)} anomalies found")
    return {
        "coin_id": coin_id,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "price_history": data["price_history"],
    }
