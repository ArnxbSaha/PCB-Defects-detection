This repository features the backend of a Machine Learning application, focusing on detecting defects in PCB images. \
The detection model used in this case is trained on a YOLOv5 model, using the dataset: **https://www.kaggle.com/datasets/akhatova/pcb-defects/data**\

In a gist, the **objectives** met by this project include:\
-Train an object detection model to detect different PCB defects using a public dataset.\
-Accept input images through an API and return predictions through bounding boxes and labels.\
-Provide visualized output with bounding boxes drawn on the images.\
-Store the prediction results in a PostgreSQL database.\

**Features:**
-Machine Learning Model: Trained on a public dataset using a YOLO-based object detection model.
-Backend: Built using FastAPI to expose the following endpoints:
          **POST /predict:**
          (POST): Accept image data and confidence limit, return predictions(bounding boxes and labels) as structured data.
          Request: Image file (multipart/form-data)
                  Confidence limit (float, optional)
          Response: List of predictions with bounding boxes and their confidence scores.
          **POST /visualize**:
          Accepts image data and a confidence limit, returning an image with drawn bounding boxes and labels.
           Request: Image file (multipart/form-data)
                  Confidence limit (float, optional)
          Response: Image with bounding boxes drawn around detected defects.
          
-Database Integration: PostgreSQL stores the results of the predictions for convenient retrieval and logging.

**Installation:**
To get this running locally, make sure you meet these prerequisites:
1. Python 3.8+
2. PostgreSQL (can be opted out of, if storing results is not a priority)

**Setup:**
1. Clone this repository:
   git clone https://github.com/ArnxbSaha/PCB-Defects-detection.git
   cd PCB-Defects-detection

2. Create a virtual environment(if needed), and install the dependencies for the project:
   python -m venv venv
   venv/bin/activate
   pip install -r requirements.txt
   
3. Training your own model:
   If you wish to train YOLOv5 according to new parameters, you can make use of the 'Dataprocess.py' and 'testing.py' codes;
   - 'Dataprocess.py' essentially does the required data preprocessing and makes a new dataset format appropriate for the YOLO format;
     The code resizes the images and splits them into training, validation and testing folders.
   - 'Testing,py' is a simple testing code to make sure the trained model is working, before moving onto deploying the model onto a backend.
   - Whilst using these codes, make sure to change the file directories to their suitable locations on your device.
     
4. Setting up backend using FASTAPI (& PostgreSQL):
   Make sure you have your trained model by this point, or you are welcome to use my pretrained model- 'best.pt'
   - Here, depending on if one wishes to integrate PostgreSQL or wants to opt out, I have provided two codes;
   - The 'main.py' integrates the trained model onto the backend using FASTAPI, with two main endpoints - /predict and /visualize, and the stores feature in a database, such as-
       1. Confidence Limit
       2. Dimensions of Bounding Boxes
       3. Image ID
   - Before running 'main.py', make sure PostgreSQL has been set up, and create a new database, either through the command line or pgAdmin4;
       -Using the command Line:
         1. psql -U postgres (assuming you are under username 'postgres')
         2. enter your password
         3. CREATE DATABASE pcb_predict;
    - After creating the database, make sure all the file directories and variable names are in order, and then use this line in the cmd line;
      uvicorn main:app --reload
    - Once the server is up, you should be able to interact with the endpoints at: **http://127.0.0.1:8000/docs**
      
    -  If you opt out of PostgreSQL, use the code 'app.py', and the rest of the steps should be the same skipping the database setting up;
      Use: uvicorn app:app --reload

 5. Access the stored data:
    From the command line, use the below-
      psql -U postgres -d pcb_predict
      enter your password
      SELECT * FROM predictions;


   

   

   
