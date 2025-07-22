# 🧠 FaceGuard – Blink-Based Mask-Supporting Face Attendance System

## Project Title / Headline  
**FaceGuard: A Real-Time Blink-Verified Attendance System Supporting Masked Face Recognition**  
An AI-powered attendance management system that supports masked faces and uses blink detection to prevent spoofing.

## 📌 Short Description / Purpose  
This project enables secure and accurate face recognition for attendance in educational institutions, even when users are wearing masks. It incorporates blink detection for liveness verification and uses facial embeddings (full and upper-face) to ensure reliability.

## 🧰 Tech Stack  
The system utilizes the following technologies:

- 🐍 **Python 3.11+** – Core logic and API development  
- ⚙️ **FastAPI** – High-performance backend framework  
- 🎥 **OpenCV** – Image processing and video frame handling  
- 🧠 **MediaPipe** – Facial landmark detection and blink analysis  
- 🔍 **Facenet (InceptionResnetV1)** – Face embedding extraction  
- 🗄️ **PostgreSQL** – Embedding storage and metadata  
- 🧪 **SQLAlchemy** – ORM for database operations  

## 📂 Data Source  
- 📸 **Input**: Real-time images captured via webcam or uploaded through API  
- 🧬 **Output**: Full and upper facial embeddings stored in PostgreSQL for identity verification  

## ✨ Features / Highlights  

### Business Problem  
In face recognition-based systems, challenges arise when users wear masks or attempt spoofing using photos. This system solves both issues using upper-face embeddings and blink detection for secure, real-world use in institutions.

### Core Features  
- 😷 **Masked Face Recognition**: Uses upper facial features (eyes, nose, ears) to identify users with masks.  
- 🧠 **Blink Detection**: Ensures the person is live before attendance is marked.  
- 🧬 **Dual Embedding Storage**: Stores both full and upper face embeddings for flexible recognition.  
- 🎓 **Subject-wise Attendance**: Tracks attendance per subject, semester, and department.  
- 🗃️ **Database Integration**: Efficient embedding and metadata storage using PostgreSQL and SQLAlchemy.  
- 🧾 **Secure API**: Built with FastAPI to handle registration and attendance seamlessly.  

### Sample Flow  
> **1. Registration**:  
> - User uploads image  
> - System stores both full and upper-face embeddings  
>  
> **2. Attendance Marking**:  
> - User faces camera  
> - Blink is verified  
> - Face is matched against database  
> - Attendance is marked for the matched subject and time  

## 📈 Business Impact & Use Cases  
- 🏫 **Colleges & Universities**: Secure, contactless, and mask-supporting attendance system  
- 🧑‍🏫 **Classroom Automation**: Reduces manual errors in attendance  
- 💼 **Workplaces**: Extendable to employee check-in systems with liveness checks  
- 🔐 **Security-Oriented Applications**: Prevents photo/spoof attacks with real-time blink verification  

## 📌 About  
FaceGuard combines modern deep learning models with facial landmarking and liveness detection to build a more secure and practical face-based attendance system in environments where masks or spoofing are common.

## 🛠️ Future Improvements  
- Add real-time webcam dashboard with live feedback  
- Support for multiple blink challenges (e.g., left eye, right eye)  
- Integration with frontend (React, Flutter)  
- Admin panel for managing users, subjects, and reports  

© 2025 | Gurleen Saini  
