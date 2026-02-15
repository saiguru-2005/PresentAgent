import os
import fitz  # PyMuPDF
from openai import OpenAI
import json

class PaperSelector:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model = "llama3.2"

    def extract_abstract(self, pdf_path):
        """Extracts text from the first page of a PDF (heuristic for Abstract)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            # Research papers usually have abstract on page 0 or 1
            for i in range(min(2, len(doc))):
                text += doc[i].get_text()
            return text[:2000] # Limit context window, abstract is usually at start
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return None

    def select_best_paper(self, pdf_paths):
        """
        Takes a list of PDF paths.
        Returns the path of the 'best' paper for a conference.
        """
        abstracts = {}
        for path in pdf_paths:
            content = self.extract_abstract(path)
            if content:
                filename = os.path.basename(path)
                abstracts[filename] = content

        if not abstracts:
            return None

        # LLM-as-a-Judge Prompt
        prompt = f"""
        You are a generic Technical Conference Chair.
        Your task is to select the ONE best paper for a presentation from the candidates below.
        
        Selection Criteria:
        1. Novelty (Is it new?)
        2. Clarity (Is it easy to present?)
        3. Impact (Does it matter?)

        Here are the abstracts:
        {json.dumps(abstracts, indent=2)}

        Analyze them briefly.
        Then, output the JSON object of the winner EXACTLY like this:
        {{
            "winner_filename": "filename.pdf",
            "reason": "Why it won..."
        }}
        Output ONLY the JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            # Clean possible markdown formatting
            content = content.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(content)
            winner_name = result.get("winner_filename")
            
            # Match back to full path
            for path in pdf_paths:
                if os.path.basename(path) == winner_name:
                    return {"path": path, "reason": result.get("reason")}
            
            # Fallback if name mismatch: return first
            return {"path": pdf_paths[0], "reason": "Fallback selection (LLM naming mismatch)"}

        except Exception as e:
            print(f"Selector Error: {e}")
            # Fallback
            return {"path": pdf_paths[0], "reason": "Fallback due to error"}

if __name__ == "__main__":
    # Test stub
    print("Selector module ready.")
