import base64
from pathlib import Path
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

def encode_image(image_path: Path) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_llm_prompt(data: Dict[str, Any]) -> ChatPromptTemplate:
    """Prepare a prompt template for the LLM"""
    
    # Create a system message that explains the task
    system_template = """
    You are an expert food measurement system that analyzes meal images to determine the weight of different food items.
    
    Your task is to:
    1. Analyze before and after images of a meal
    2. Use reference product images with known weights to estimate the weight of each product in the meal
    3. Determine how much of each product was consumed
    4. Return the analysis in a structured JSON format
    
    Remember:
    - You will see the "before" and "after" images of the complete meal
    - For each product, you will see reference images of that product on a plate with known weight
    - Use the plate dimensions and visual cues to estimate portion sizes
    - The meal total weight before was {weight_before}g 
    - Your estimates should account for the full weight of the meal
    
    IMPORTANT: Return your analysis in this exact JSON structure:
    ```
    {{
      "total_estimated_weight_before": float,
      "total_estimated_weight_after": float,
      "total_estimated_consumed": float,
      "products_analysis": [
        {{
          "sku": "string",
          "name": "string",
          "estimated_weight_before": float,
          "estimated_weight_after": float,
          "estimated_consumed": float,
          "confidence": int
        }}
      ],
      "notes": "string"
    }}
    ```
    
    Provide all measurements in grams and confidence levels as integers from 1-100.
    """
    
    # Create a human message template with the images and data
    human_template = """
    ## Meal Information
    - Description: {description}
    - Total weight before: {weight_before}g
    
    ## Meal Images
    Before meal image:
    <img src="data:image/jpeg;base64,{picture_before_base64}" width="400" />
    
    After meal image:
    <img src="data:image/jpeg;base64,{picture_after_base64}" width="400" />
    
    ## Products to Analyze
    {products_info}
    
    Please analyze the images and provide weight estimates for each product before and after the meal.
    Return your response in the structured JSON format as specified.
    """
    
    # Format product information with reference images
    products_info = ""
    for product in data["products"]:
        products_info += f"### {product['name']} (SKU: {product['sku']})\n"
        
        for picture in product["pictures"]:
            products_info += f"Reference image (weight: {picture['weight']}g, plate ID: {picture['plateId']}):\n"
            products_info += f"<img src=\"data:image/jpeg;base64,{picture['image_base64']}\" width=\"300\" />\n"
            products_info += f"Plate dimensions: upper diameter: {picture['upperDiameter']}cm, " \
                           f"lower diameter: {picture['lowerDiameter']}cm, depth: {picture['depth']}cm\n\n"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    # Format the variables in the prompt
    formatted_prompt = prompt.partial(
        weight_before=data["weight_before"],
        description=data["description"],
        picture_before_base64=data["picture_before_base64"],
        picture_after_base64=data["picture_after_base64"],
        products_info=products_info
    )
    
    return formatted_prompt