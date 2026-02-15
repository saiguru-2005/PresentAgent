from pptagent.llms import AsyncLLM
import logging
import json

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, language_model: AsyncLLM, vision_model: AsyncLLM):
        self.language_model = language_model
        self.vision_model = vision_model

    async def evaluate_content(self, source_text: str, presentation_content: str):
        prompt = f"""
        You are an expert Presentation Grader.
        
        Original Document (Summary):
        {source_text[:2000]}...

        Generated Slides Content:
        {presentation_content}

        Task: Rate the CONTENT QUALITY (1-10).
        - Does it cover the key points?
        - Is it accurate?
        - Is the logical flow good?

        Return valid JSON:
        {{
            "score": 8,
            "reason": "..."
        }}
        """
        try:
            response = await self.language_model(prompt, return_json=True)
            return response
        except Exception as e:
            logger.error(f"Content Eval failed: {e}")
            return {"score": 0, "error": str(e)}

    async def evaluate_visuals(self, slide_image_paths: list[str]):
        if not slide_image_paths:
            return {"score": 0, "reason": "No images generated."}
        
        # Checking just the first 3 slides to save bandwidth/time
        images_to_check = slide_image_paths[:3] 
        
        prompt = """
        You are an expert Design Critic.
        Rate the VISUAL QUALITY (1-10) of these presentation slides.
        - Are they readable?
        - Is the layout professional?
        - Do the images match the context?

        Return valid JSON:
        {{
            "score": 7,
            "reason": "..."
        }}
        """
        try:
            # Note: This assumes the vision model accepts a list of image paths
            # If the underlying AsyncLLM implementation expects one image at a time, loop might be needed.
            # But standard Gemini API supports multiple images.
            response = await self.vision_model(prompt, images=images_to_check, return_json=True)
            return response
        except Exception as e:
            logger.error(f"Visual Eval failed: {e}")
            return {"score": 0, "error": str(e)}

    async def generate_quiz(self, source_text: str):
        prompt = f"""
        Generate a 5-question multiple choice quiz based on this text to test comprehension.
        
        Text:
        {source_text[:3000]}...

        Return valid JSON list:
        [
            {{
                "question": "...",
                "options": ["a", "b", "c", "d"],
                "answer": "a"
            }}
        ]
        """
        try:
            response = await self.language_model(prompt, return_json=True)
            return response
        except Exception as e:
            logger.error(f"Quiz Gen failed: {e}")
            return []
