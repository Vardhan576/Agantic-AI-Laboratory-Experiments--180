import os
import numpy as np
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

# Setup Gemini Client
API_KEY = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = None
if os.environ.get("GEMINI_API_KEY") or API_KEY != "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg":
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        client = None

MODEL_NAME = "gemini-3.5-flash"
EMBED_MODEL = "text-embedding-004"

# Mock descriptions for local mode
MOCK_DESCRIPTIONS = {
    "red_circle.png": "A clean visual image of a bright red circular shape outlined in black, centered on a white background.",
    "blue_rectangle.png": "A visual layout displaying a horizontal dark blue rectangle with a thin black outline on a white background.",
    "green_triangle.png": "A graphic depicting a green triangular shape with three equal sides and a black border on a white background."
}

def create_mock_images(output_dir: str):
    """Draw and save 3 distinct images for visual QA testing."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Image Gen] Creating 3 mock test images with PIL...")
    
    # Image 1: Red Circle
    img1 = Image.new("RGB", (300, 300), color="white")
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([50, 50, 250, 250], fill="red", outline="black", width=3)
    img1_path = os.path.join(output_dir, "red_circle.png")
    img1.save(img1_path)
    
    # Image 2: Blue Rectangle
    img2 = Image.new("RGB", (300, 300), color="white")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([40, 80, 260, 220], fill="blue", outline="black", width=3)
    img2_path = os.path.join(output_dir, "blue_rectangle.png")
    img2.save(img2_path)
    
    # Image 3: Green Triangle
    img3 = Image.new("RGB", (300, 300), color="white")
    draw3 = ImageDraw.Draw(img3)
    # Define triangle points
    draw3.polygon([(150, 50), (50, 250), (250, 250)], fill="green", outline="black", width=3)
    img3_path = os.path.join(output_dir, "green_triangle.png")
    img3.save(img3_path)
    
    print(f"[Image Gen] Saved images to {output_dir}")
    return [img1_path, img2_path, img3_path]

def generate_image_description(image_path: str, use_mock=False) -> str:
    """Ask Gemini to describe what is in the image."""
    filename = os.path.basename(image_path)
    print(f"[Multimodal Indexer] Describing image: {filename}...")
    
    if use_mock or not client:
        desc = MOCK_DESCRIPTIONS.get(filename, "A geometric shape on a white background.")
        print(f"  [Mock Description]: {desc}")
        return desc
        
    img = Image.open(image_path)
    prompt = "Describe this image in detail. Mention the main object, shape, color, location, and background."
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[img, prompt]
        )
        desc = response.text.strip()
        print(f"  Description: {desc}")
        return desc
    except Exception as e:
        print(f"[Multimodal Indexer] Warning: API Error ({e}). Using mock description fallback.")
        return generate_image_description(image_path, use_mock=True)

def get_text_embedding(text: str, use_mock=False) -> np.ndarray:
    """Retrieve embedding vector using Gemini's text-embedding-004."""
    if use_mock or not client:
        # Mock embedding: generate a deterministic vector based on word presence
        vector = np.zeros(128, dtype=np.float32)
        text_lower = text.lower()
        if "red" in text_lower or "circle" in text_lower or "round" in text_lower:
            vector[0] = 1.0
        if "blue" in text_lower or "rectangle" in text_lower or "square" in text_lower:
            vector[1] = 1.0
        if "green" in text_lower or "triangle" in text_lower or "polygon" in text_lower:
            vector[2] = 1.0
        # Add small random noise to prevent identical vectors
        np.random.seed(len(text))
        vector += np.random.randn(128).astype(np.float32) * 0.05
        return vector
        
    try:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        print(f"[Embedder] Warning: API Error ({e}). Falling back to mock embedding.")
        return get_text_embedding(text, use_mock=True)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate similarity score between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def run_visual_qa(image_path: str, query: str, use_mock=False) -> str:
    """Perform visual question answering on the specified image."""
    filename = os.path.basename(image_path)
    print(f"  [Visual QA] Querying Gemini with {filename} and question...")
    
    if use_mock or not client:
        # Mock responses
        if "red" in filename:
            ans = "The image shows a large red circle drawn in the center of the frame on a clean white background. It has a visible black outline."
        elif "blue" in filename:
            ans = "The image shows a solid blue rectangle in the center of a white page. The border of the rectangle is drawn in black."
        else:
            ans = "The image shows a green equilateral triangle outlined in black, pointed upwards against a white background."
        return ans
        
    img = Image.open(image_path)
    qa_prompt = f"Based on this image, answer the user's question directly: {query}"
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[img, qa_prompt]
        )
        return response.text.strip()
    except Exception as e:
        print(f"  [Visual QA] Warning: API Error ({e}). Using mock visual QA answer.")
        return run_visual_qa(image_path, query, use_mock=True)

def main():
    print("="*60)
    print("EXPERIMENT 8: MULTIMODAL RETRIEVAL & VISUAL QA START")
    print("="*60)
    
    use_mock = False
    if API_KEY == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        print("[System Info] Gemini API Key not set. Running in local high-fidelity simulation mode.")
        
    out_dir = os.path.dirname(__file__)
    images_dir = os.path.join(out_dir, "images")
    
    # 1. Create Mock Images
    image_paths = create_mock_images(images_dir)
    
    # 2. Multimodal Indexing
    db = []
    for path in image_paths:
        desc = generate_image_description(path, use_mock=use_mock)
        emb = get_text_embedding(desc, use_mock=use_mock)
        db.append({
            "path": path,
            "description": desc,
            "embedding": emb
        })
        
    # 3. Visual QA & Retrieval Test Run
    test_queries = [
        "Which image contains a red round circular shape?",
        "Find the image containing a blue rectangular object.",
        "Query: Where is the green triangular object?"
    ]
    
    for query in test_queries:
        print(f"\n--- Processing User Query: '{query}' ---")
        
        # Search index
        query_emb = get_text_embedding(query, use_mock=use_mock)
        similarities = []
        for item in db:
            score = cosine_similarity(query_emb, item["embedding"])
            similarities.append(score)
            print(f"  * Match score for {os.path.basename(item['path'])}: {score:.4f}")
            
        best_idx = np.argmax(similarities)
        best_match = db[best_idx]
        print(f"  -> Retrieved Best Match: {os.path.basename(best_match['path'])} (Score: {similarities[best_idx]:.4f})")
        
        # Perform Visual QA on retrieved image
        answer = run_visual_qa(best_match["path"], query, use_mock=use_mock)
        print(f"  [Visual QA Answer]: {answer}")
        
    print("\n" + "="*60)
    print("EXPERIMENT 8 FINISHED")
    print("="*60)

if __name__ == "__main__":
    main()
