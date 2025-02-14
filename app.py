# backend/main.py
from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from litellm import completion
import os
import dotenv
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Load environment variables
dotenv.load_dotenv()

# Set up Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY") 
ports=os.getenv("PORT")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://memes-alpha-two.vercel.app/"],  # Replace with your Vercel URL
    #allow_origins=["*"],  # Replace "*" with ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class CaptionRequest(BaseModel):
    text: str

def replace_figure(template_path, face_path, output_path, position):
    # Load the template image
    template = cv2.imread(template_path)
    
    # Load the face image
    face = cv2.imread(face_path, -1)  # -1 to load with alpha channel if available
    
    # Resize the face image to fit the template (adjust size as needed)
    face = cv2.resize(face, (100, 100))  # Example size, adjust as needed
    
    # If the face image has an alpha channel, use it as a mask
    if face.shape[2] == 4:
        alpha = face[:, :, 3] / 255.0
        face = face[:, :, :3]
    else:
        alpha = np.ones(face.shape[:2], dtype=face.dtype)
    
    # Define the region of interest (ROI) in the template
    x, y = position
    h, w = face.shape[:2]
    roi = template[y:y+h, x:x+w]
    
    # Blend the face image with the ROI using the alpha mask
    for c in range(0, 3):
        roi[:, :, c] = (alpha * face[:, :, c] + (1 - alpha) * roi[:, :, c])
    
    # Save the result
    cv2.imwrite(output_path, template)



@app.post("/generate-caption")
async def generate_caption(request: CaptionRequest):
    try:
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": f"Generate a funny meme caption for: {request.text}"}],
            api_key=gemini_api_key,
        )
        caption = response.choices[0].message.content.strip()
        print("Generated Caption:", caption)  # Debugging line
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meme-templates")
async def get_meme_templates():
    return ([
        
        {"name": "Drakeposting", "url": "https://i.imgflip.com/30b1gx.jpg"},
        {"name": "Two Buttons", "url": "https://i.imgflip.com/1g8my4.jpg"},
        {"name": "Expanding Brain", "url": "https://i.imgflip.com/1jwhww.jpg"}
    ])
@app.post("/process-images/")
async def process_images(
    template: UploadFile = File(...),
    face: UploadFile = File(...),
    x: int = 50,
    y: int = 50
):
    # Save uploaded files temporarily
    template_path = "temp_template.png"
    face_path = "temp_face.png"
    output_path = "output.png"
    
    try:
        # Save the uploaded template image
        with open(template_path, "wb") as buffer:
            buffer.write(await template.read())
        
        # Save the uploaded face image
        with open(face_path, "wb") as buffer:
            buffer.write(await face.read())
        
        # Process the images using the replace_figure function
        replace_figure(template_path, face_path, output_path, (x, y))
        
        # Return the processed image
        return FileResponse(output_path, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if os.path.exists(template_path):
            os.remove(template_path)
        if os.path.exists(face_path):
            os.remove(face_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=ports)