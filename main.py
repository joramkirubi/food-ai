from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #allow access from anywhere(safe for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

@app.get("/")
def home():
    return{"message": "Food AI Backend is running 🚀"}

#Test endpoint to upload image and get its size
@app.post("/upload-image")
async def upload_image(image:UploadFile = File(...)):
    contents = await image.read() # read image bytes
    size = len(contents)
    return {"filename" : image.filename, "size_in_bytes" : size}

# 2. identify-food + ingredients
@app.post("/identify-food")
async def identify_food(image: UploadFile = File(...)):
    # Read image bytes
    img_bytes = await image.read()

    #convert to base64 string
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Ask GPT Visison
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the food item in the image and list the possible ingredients."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
    )

    analysis = response.choices[0].message.content
    return {"analysis": analysis}

# 3. generate-recipe
@app.post("/generate-recipe")
async def generate_recipe(payload: dict):
    dish_name = payload.get("dish_name", "Unknown Dish")
    ingredients_list = payload.get("ingredients", "")
    prompt = f"""
    You are a highly skilled Kenyan chef with deep expertise in traditional and modern East African cuisine.
Your task is to generate beginner-friendly recipes that are clear, accurate, and easy to follow.
You must always respect Kenyan culinary techniques, ingredient usage, and cooking traditions.

When given a dish name and a list of detected ingredients, produce a recipe in the following exact structure:

1. Title
2. Total Time Required
3. Ingredients
4. Step-by-step Instructions
5. Optional: Serving Suggestions

Guidelines you must follow:

Always estimate a realistic total cooking time (prep + cooking).

Keep explanations simple so beginners can cook the dish confidently.

Use Kenyan/East African cooking methods, flavors, and terminology where appropriate.

If ingredients are missing, sensibly add basic essentials (e.g., oil, salt, water).

Ensure steps flow logically from preparation to cooking to serving.

Be precise, culturally authentic, and helpful—never overly wordy.

Inputs you will receive:

Dish name: {dish_name}

Ingredients detected: {ingredients_list}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    recipe = response.choices[0].message.content
    return {"recipe": recipe}