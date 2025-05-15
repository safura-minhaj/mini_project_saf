Breast Cancer Prediction App
============================

Description:
------------
This is a full-stack web application for predicting breast cancer using a machine learning model. 
It consists of a Python Flask backend and a Node.js frontend.

Project Structure:
------------------
/your-project-root
├── backend/
│   ├── app.py
│   ├── venv/
│   └── requirements.txt
└── frontend/
    ├── package.json
    └── ...

Setup Instructions:
-------------------

1. Clone the Repository
-----------------------
git clone <insert path here>
cd breast-cancer-app

2. Backend Setup (Python + Flask)
---------------------------------
cd backend

a. Create and activate a virtual environment:
   - python3 -m venv venv
   - source venv/bin/activate   (On Windows: venv\Scripts\activate)

b. Install dependencies:
   - pip install -r requirements.txt

c. Run the backend server:
   - python app.py

   -> Backend will be running at: http://localhost:5000

3. Frontend Setup (Node.js)
---------------------------
Open a new terminal and navigate to the frontend folder:

cd frontend

a. Install dependencies:
   - npm install

b. Start the development server:
   - npm start

   -> Frontend will be running at: http://localhost:3000

Notes:
------
- Both backend and frontend servers should be running simultaneously.
- Do NOT upload `venv/` or `node_modules/` folders to GitHub.
- Use a `.gitignore` file to exclude unnecessary folders.
- If needed, configure CORS and environment variables using `.env` files.

License:
--------
MIT License — you are free to use and modify this project.
