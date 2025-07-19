# [All imports and setup remain unchanged as you shared]
# START OF FILE

import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Dict
from facenet_pytorch import InceptionResnetV1
import logging

# Logger setup
logger = logging.getLogger("uvicorn.error")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB setup
DATABASE_URL = "postgresql://postgres:pngalele@localhost/attendance_db"
engine = sqlalchemy.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    name = sqlalchemy.Column(sqlalchemy.String)
    roll_number = sqlalchemy.Column(sqlalchemy.String, unique=True)
    department = sqlalchemy.Column(sqlalchemy.String)
    embeddings = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.Float))
    upper_embeddings = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.Float))
    created_at = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.utcnow)

class Attendance(Base):
    __tablename__ = "attendance"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    student_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("students.id"))
    branch = sqlalchemy.Column(sqlalchemy.String)
    semester = sqlalchemy.Column(sqlalchemy.Integer)
    subject = sqlalchemy.Column(sqlalchemy.String)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Utility functions
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def extract_eyes_and_ear(landmarks, image_shape):
    h, w = image_shape[:2]
    eye_indices = {
        'left': [362, 385, 387, 263, 373, 380],
        'right': [33, 160, 158, 133, 153, 144]
    }
    eyes = {side: [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]
            for side, idxs in eye_indices.items()}
    left_ear = eye_aspect_ratio(eyes['left'])
    right_ear = eye_aspect_ratio(eyes['right'])
    return (left_ear + right_ear) / 2.0

def extract_embedding(image, landmark_indices=None):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None

    face = results.multi_face_landmarks[0]
    h, w = image.shape[:2]

    if landmark_indices is None:
        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]
    else:
        xs = [face.landmark[i].x for i in landmark_indices]
        ys = [face.landmark[i].y for i in landmark_indices]

    left = max(int(min(xs) * w) - 20, 0)
    right = min(int(max(xs) * w) + 20, w)
    top = max(int(min(ys) * h) - 20, 0)
    bottom = min(int(max(ys) * h) + 20, h)

    if top >= bottom or left >= right:
        return None, None

    face_crop = image[top:bottom, left:right]
    if face_crop.size == 0:
        return None, None

    face_crop = cv2.resize(face_crop, (160, 160))
    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    embedding = resnet(face_tensor).detach().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding, face.landmark

upper_face_indices = list(range(33, 133)) + list(range(246, 279)) + list(range(443, 464))

def recognize_face_by_embedding(embedding, db, column='embeddings', threshold=0.55):
    students = db.query(Student).all()
    best_score = -1
    matched = None
    for student in students:
        stored = getattr(student, column)
        if stored is None:
            continue
        stored = np.array(stored)
        stored = stored / np.linalg.norm(stored)
        score = np.dot(stored, embedding)
        if score > threshold and score > best_score:
            best_score = score
            matched = student
    return matched, best_score

@app.get("/")
def root():
    return {"status": "Face Attendance System with Masked Support"}


@app.post("/register")
async def register_face(
    file: UploadFile = File(...),
    x_user_name: str = Header(...),
    x_user_roll: str = Header(...),
    x_user_dept: str = Header(...),
):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    full_embedding, landmarks = extract_embedding(img)
    if full_embedding is None or landmarks is None:
        return {"registered": False, "message": "Face not detected"}

    upper_embedding, _ = extract_embedding(img, upper_face_indices)
    if upper_embedding is None:
        return {"registered": False, "message": "Upper face embedding failed"}

    db = SessionLocal()
    try:
        # ✅ Step 1: Check face similarity against existing embeddings
        student_full, sim_full = recognize_face_by_embedding(full_embedding, db, 'embeddings', threshold=0.55)
        student_upper, sim_upper = recognize_face_by_embedding(upper_embedding, db, 'upper_embeddings', threshold=0.50)

        if student_full or student_upper:
            existing = student_full if student_full else student_upper
            return {
                "registered": False,
                "message": "Face already registered",
                "user": {
                    "name": existing.name,
                    "roll_number": existing.roll_number,
                    "department": existing.department,
                },
            }

        # ✅ Step 2: Check duplicate roll number (acts as unique identifier)
        existing_roll = db.query(Student).filter_by(roll_number=x_user_roll).first()
        if existing_roll:
            return {
                "registered": False,
                "message": "User already registered with this roll number",
                "user": {
                    "name": existing_roll.name,
                    "roll_number": existing_roll.roll_number,
                    "department": existing_roll.department,
                },
            }

        # ✅ Step 3: Save new student
        student = Student(
            name=x_user_name,
            roll_number=x_user_roll,
            department=x_user_dept,
            embeddings=full_embedding.tolist(),
            upper_embeddings=upper_embedding.tolist()
        )
        db.add(student)
        db.commit()

        return {
            "registered": True,
            "message": "User registered successfully",
            "user": {
                "name": student.name,
                "roll_number": student.roll_number,
                "department": student.department,
            },
        }
    finally:
        db.close()


@app.post("/recognize")
async def recognize_face_endpoint(
    file: UploadFile = File(...),
    x_branch: str = Header(...),
    x_semester: int = Header(...),
    x_subject: str = Header(...),
):
    valid_subjects = {
        1: {"maths", "physics", "chemistry", "c programming", "os"},
        2: {"dbms", "web", "oops", "python", "java"},
    }

    if x_semester not in valid_subjects or x_subject.lower() not in valid_subjects[x_semester]:
        raise HTTPException(status_code=400, detail="Invalid semester or subject")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    full_embedding, _ = extract_embedding(img)
    upper_embedding, _ = extract_embedding(img, upper_face_indices)

    if full_embedding is None and upper_embedding is None:
        return {"recognized": False, "message": "No face detected"}

    db = SessionLocal()
    try:
        student, similarity = None, None
        if full_embedding is not None:
            student, similarity = recognize_face_by_embedding(full_embedding, db, 'embeddings')
        if not student and upper_embedding is not None:
            student, similarity = recognize_face_by_embedding(upper_embedding, db, 'upper_embeddings', threshold=0.50)

        if not student:
            return {"recognized": False, "message": "Face not recognized"}

        # ✅ Check if branch matches registration
        if student.department.lower() != x_branch.lower():
            return {
                "recognized": False,
                "message": f"Branch mismatch! Registered: {student.department}, Selected: {x_branch}"
            }

        return {
            "recognized": True,
            "message": "User recognized",
            "similarity": round(similarity, 3),
            "user": {
                "id": student.id,
                "name": student.name,
                "roll_number": student.roll_number,
                "department": student.department,
            },
        }
    finally:
        db.close()


# Blink management
class BlinkDetector:
    def __init__(self, threshold=0.21):
        self.threshold = threshold
        self.blinked = False

    def update(self, ear):
        if not self.blinked and ear < self.threshold:
            self.blinked = True
            return True
        return False

class BlinkManager:
    def __init__(self):
        self.detectors: Dict[str, BlinkDetector] = {}

    def update(self, user_id: str, ear: float) -> bool:
        if user_id not in self.detectors:
            self.detectors[user_id] = BlinkDetector()
        return self.detectors[user_id].update(ear)

    def reset(self, user_id: str):
        self.detectors.pop(user_id, None)

blink_manager = BlinkManager()

@app.post("/blink_challenge")
async def blink_challenge(
    user_id: str,
    file: UploadFile = File(...),
    x_branch: str = Header(...),
    x_semester: int = Header(...),
    x_subject: str = Header(...),
):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    _, landmarks = extract_embedding(img)
    if landmarks is None:
        return {"blinked": False}

    ear = extract_eyes_and_ear(landmarks, img.shape)
    blinked = blink_manager.update(user_id, ear)

    if blinked:
        db = SessionLocal()
        try:
            student = db.query(Student).filter_by(id=int(user_id)).first()
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")

            # ✅ Check if branch matches again
            if student.department.lower() != x_branch.lower():
                blink_manager.reset(user_id)
                return {
                    "blinked": False,
                    "message": f"Branch mismatch! Registered: {student.department}, Selected: {x_branch}"
                }

            today = datetime.utcnow().date()
            already_marked = db.query(Attendance).filter(
                Attendance.student_id == student.id,
                Attendance.branch == x_branch,
                Attendance.semester == x_semester,
                Attendance.subject == x_subject.lower(),
                Attendance.timestamp >= datetime(today.year, today.month, today.day)
            ).first()

            if already_marked:
                blink_manager.reset(user_id)
                return {
                    "blinked": False,
                    "message": "Attendance already marked"
                }

            db.add(Attendance(
                student_id=student.id,
                branch=x_branch,
                semester=x_semester,
                subject=x_subject.lower()
            ))
            db.commit()
            blink_manager.reset(user_id)
            return {
                "blinked": True,
                "message": "Attendance recorded",
                "user": {
                    "name": student.name,
                    "roll_number": student.roll_number,
                    "department": student.department,
                },
            }
        finally:
            db.close()

    return {"blinked": False}
