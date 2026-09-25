import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./requests.db"

# check_same_thread needed since fastapi hits this from multiple threads
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_log"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    endpoint = Column(String)
    status = Column(String)  # success / failed
    detail = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_request(db, ticker: str, endpoint: str, status: str, detail: str = None):
    # just a simple audit trail, nothing fancy
    entry = RequestLog(ticker=ticker, endpoint=endpoint, status=status, detail=detail)
    db.add(entry)
    db.commit()
