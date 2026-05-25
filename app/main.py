import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from openai import AsyncOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = AsyncOpenAI(api_key=api_key)

app = FastAPI(title="Job Search Assistant")


class JobRequest(BaseModel):
    job_description: str


@app.get("/")
async def read_root():
    return {
        "message": "Job Search AI Assistant API",
        "status": "healthy",
        "endpoints": {
            "analyze": "/analyze",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    }


@app.post("/analyze")
async def analyze_job(request: JobRequest):
    try:
        system_prompt = """
You are an expert career coach and technical recruiter with 10+ years experience.
Your job is to analyze job descriptions for job seekers.
Always respond with valid JSON only. No extra text.
"""
        user_prompt = f"""
Analyze the following job description.

Return a JSON object with exactly this structure:
{{
  "role_summary": "One short sentence describing the role",
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "resume_focus_areas": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "key_responsibilities": ["responsibility 1", "responsibility 2"]
}}

Job Description:
{request.job_description}
"""
        # calling the OpenAI API
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content.strip()

        # Guardrail

        parsed_json = json.loads(raw_content)
        required_keys = {"role_summary", "required_skills", "preferred_skills", "resume_focus_areas", "key_responsibilities"}

        for key in required_keys:
            if key not in parsed_json:
                parsed_json[key] = [] if "skills" in key or "areas" in key or "responsibilities" in key else "Not provided"

        return {"analysis": parsed_json}

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from OpenAI response", "raw_response": raw_content}

    except Exception as e:
        print("ERROR in analyze_job:", str(e))
        raise HTTPException(status_code=500, detail=str(e))