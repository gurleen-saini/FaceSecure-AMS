import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [status, setStatus] = useState('Ready');
  const [detectedUser, setDetectedUser] = useState(null);
  const [registerMode, setRegisterMode] = useState(false);
  const [name, setName] = useState('');
  const [roll, setRoll] = useState('');
  const [dept, setDept] = useState('CSE');
  const [blinkInProgress, setBlinkInProgress] = useState(false);

  const capture = async () => {
    setStatus('Processing...');
    const imageSrc = webcamRef.current.getScreenshot();
    const blob = await fetch(imageSrc).then(res => res.blob());
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    try {
      if (registerMode) {
        const response = await axios.post('http://localhost:8000/register', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            'X-User-Name': name,
            'X-User-Roll': roll,
            'X-User-Dept': dept,
          },
        });
        setStatus(`Registered: ${response.data.name}`);
        setRegisterMode(false);
        setName('');
        setRoll('');
        setDept('CSE');
      } else {
        const response = await axios.post('http://localhost:8000/recognize', formData);
        setDetectedUser(response.data.user);
        setStatus(response.data.message);

        if (!response.data.already_marked) {
          setBlinkInProgress(true);
          startBlinkDetection(response.data.user.id);
        }
      }
    } catch (error) {
      setStatus(error.response?.data?.detail || 'Error occurred');
    }
  };

  const startBlinkDetection = (userId) => {
    let attempts = 0;
    const maxAttempts = 25;

    const intervalId = setInterval(async () => {
      if (attempts >= maxAttempts) {
        clearInterval(intervalId);
        setBlinkInProgress(false);
        setStatus('Blink timeout: Try again.');
        return;
      }

      const imageSrc = webcamRef.current.getScreenshot();
      const blob = await fetch(imageSrc).then(res => res.blob());
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      try {
        const response = await axios.post(`http://localhost:8000/blink_challenge?user_id=${userId}`, formData);
        if (response.data.blinked) {
          clearInterval(intervalId);
          setBlinkInProgress(false);
          setStatus(response.data.message);
        }
      } catch (error) {
        clearInterval(intervalId);
        setBlinkInProgress(false);
        setStatus('Blink challenge failed.');
      }

      attempts++;
    }, 500);
  };

  return (
    <div className="app">
      <h1>Face Attendance System</h1>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: 'user' }}
      />

      {registerMode && (
        <div className="registration-form">
          <input type="text" placeholder="Name" value={name} onChange={e => setName(e.target.value)} />
          <input type="text" placeholder="Roll Number" value={roll} onChange={e => setRoll(e.target.value)} />
          <select value={dept} onChange={e => setDept(e.target.value)}>
            <option value="CSE">CSE</option>
            <option value="IT">IT</option>
            <option value="EEE">EEE</option>
            <option value="ECE">ECE</option>
            <option value="MECH">MECH</option>
          </select>
        </div>
      )}

      <div className="controls">
        <button onClick={capture} disabled={blinkInProgress}>
          {registerMode ? 'Register Face' : 'Mark Attendance'}
        </button>
        <button onClick={() => setRegisterMode(!registerMode)} disabled={blinkInProgress}>
          {registerMode ? 'Cancel' : 'New Registration'}
        </button>
      </div>

      <div className="status">
        <p>{status}</p>
        {detectedUser && (
          <div className="user-info">
            <p>Name: {detectedUser.name}</p>
            <p>Roll: {detectedUser.roll_number}</p>
            <p>Dept: {detectedUser.department}</p>
            <p>Time: {new Date().toLocaleTimeString()}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
