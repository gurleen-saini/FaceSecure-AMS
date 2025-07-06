# main.py
import os
import cv2
import dlib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional, Dict
from collections import deque

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "postgresql://postgres:pngalele@localhost/attendance_db"
engine = sqlalchemy.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    name = sqlalchemy.Column(sqlalchemy.String, index=True)
    roll_number = sqlalchemy.Column(sqlalchemy.String, unique=True, nullable=False)
    department = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    embeddings = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.Float))
    created_at = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.utcnow)

class Attendance(Base):
    __tablename__ = "attendance"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    student_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("students.id"))
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

class BlinkDetector:
    def __init__(self, ear_threshold=0.21):
        self.ear_threshold = ear_threshold
        self.blinked = False

    def update(self, ear: float) -> bool:
        print(f"[DEBUG] EAR: {ear:.3f}")
        if not self.blinked and ear < self.ear_threshold:
            self.blinked = True
            print("[INFO] Blink detected!")
            return True
        return False


class BlinkManager:
    def __init__(self):
        self.detectors: Dict[str, BlinkDetector] = {}

    def update_user(self, user_id: str, ear: float) -> bool:
        if user_id not in self.detectors:
            self.detectors[user_id] = BlinkDetector()
        return self.detectors[user_id].update(ear)

    def reset_user(self, user_id: str):
        if user_id in self.detectors:
            del self.detectors[user_id]

blink_manager = BlinkManager()

def get_eye_aspect_ratio(eye_points):
    v1 = dist(eye_points[1], eye_points[5])
    v2 = dist(eye_points[2], eye_points[4])
    h = dist(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def dist(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

def get_face_embeddings(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None, None
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, face)
    embeddings = face_rec_model.compute_face_descriptor(frame, landmarks)
    return np.array(embeddings), (gray, landmarks, face)

def extract_ear(gray, landmarks):
    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    return (get_eye_aspect_ratio(left_eye) + get_eye_aspect_ratio(right_eye)) / 2.0

def recognize_face(embeddings, db):
    students = db.query(Student).all()
    min_dist = float('inf')
    recognized_student = None
    for student in students:
        dist = np.linalg.norm(np.array(student.embeddings) - embeddings)
        if dist < 0.6 and dist < min_dist:
            min_dist = dist
            recognized_student = student
    return recognized_student
@app.post("/register")
async def register_face(
    file: UploadFile = File(...),
    x_user_name: str = Header(...),
    x_user_roll: str = Header(...),
    x_user_dept: str = Header(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    embeddings, _ = get_face_embeddings(frame)

    if embeddings is None:
        raise HTTPException(status_code=400, detail="No face detected")

    with SessionLocal() as db:
        existing = db.query(Student).filter_by(roll_number=x_user_roll).first()
        if existing:
            raise HTTPException(status_code=409, detail="User already registered")

        student = Student(
            name=x_user_name,
            roll_number=x_user_roll,
            department=x_user_dept,
            embeddings=embeddings.tolist()
        )
        db.add(student)
        db.commit()
        db.refresh(student)

        return {
            "id": student.id,
            "name": student.name,
            "roll_number": student.roll_number,
            "department": student.department
        }

@app.post("/recognize")
async def recognize_face_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    embeddings, _ = get_face_embeddings(frame)
    if embeddings is None:
        raise HTTPException(status_code=400, detail="No face detected")

    with SessionLocal() as db:
        student = recognize_face(embeddings, db)
        if student is None:
            raise HTTPException(status_code=404, detail="User not recognized")

        today = datetime.utcnow().date()
        existing = db.query(Attendance).filter(
            Attendance.student_id == student.id,
            Attendance.timestamp >= datetime(today.year, today.month, today.day)
        ).first()

        if existing:
            return {
                "message": "Attendance already marked",
                "user": {
                    "id": student.id,
                    "name": student.name,
                    "roll_number": student.roll_number,
                    "department": student.department
                },
                "already_marked": True
            }

        return {
            "message": "User recognized. Please blink to confirm attendance.",
            "user": {
                "id": student.id,
                "name": student.name,
                "roll_number": student.roll_number,
                "department": student.department
            },
            "already_marked": False
        }

@app.post("/blink_challenge")
async def blink_challenge(user_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return {"blinked": False}

    for face in faces:
        landmarks = predictor(gray, face)
        ear = extract_ear(gray, landmarks)
        blinked = blink_manager.update_user(user_id, ear)
        if blinked:
            with SessionLocal() as db:
                student = db.query(Student).filter(Student.id == int(user_id)).first()
                if not student:
                    raise HTTPException(status_code=404, detail="Student not found")

                attendance = Attendance(student_id=student.id)
                db.add(attendance)
                db.commit()

                blink_manager.reset_user(user_id)

                return {"blinked": True, "message": "Attendance recorded", "user": {"name": student.name, "roll_number": student.roll_number, "department": student.department}}

    return {"blinked": False}

@app.get("/")
def read_root():
    return {"status": "Face Attendance System with Blink Challenge is running"}
