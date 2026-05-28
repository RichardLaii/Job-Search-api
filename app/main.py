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

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

client = AsyncOpenAI(api_key=api_key)

app = FastAPI(title="Job Search Assistant")


class JobRequest(BaseModel):
    job_description: str


class TailorResumeRequest(BaseModel):
    job_description: str
    resume_text: str


DEFAULT_ANALYSIS = {
    "role_summary": "Not provided",
    "required_skills": [],
    "preferred_skills": [],
    "resume_focus_areas": [],
    "key_responsibilities": [],
}

DEFAULT_TAILORED = {
    "match_summary": "Not provided",
    "missing_keywords": [],
    "rewrite_suggestions": [],
    "skills_section_suggestion": [],
    "warnings": [],
}


def _apply_defaults(data: dict, defaults: dict) -> dict:
    for key, default in defaults.items():
        if key not in data:
            data[key] = default
    return data


async def _openai_json(system_prompt: str, user_prompt: str) -> dict:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )

    if not getattr(response, "choices", None):
        raise HTTPException(status_code=502, detail="OpenAI returned no choices")

    message = response.choices[0].message
    content = getattr(message, "content", None)
    if content is None:
        raise HTTPException(status_code=502, detail="OpenAI returned no content")

    raw_content = content.strip()
    if not raw_content:
        raise HTTPException(status_code=502, detail="OpenAI returned empty content")

    try:
        parsed_json = json.loads(raw_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Failed to parse JSON from OpenAI response")

    if not isinstance(parsed_json, dict):
        raise HTTPException(status_code=502, detail="OpenAI returned non-object JSON")

    return parsed_json


@app.get("/")
async def read_root():
    return {
        "message": "Job Search AI Assistant API",
        "status": "healthy",
        "endpoints": {
            "analyze": "/analyze",
            "tailor_resume": "/tailor_resume",
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
        parsed_json = await _openai_json(system_prompt=system_prompt, user_prompt=user_prompt)
        _apply_defaults(parsed_json, DEFAULT_ANALYSIS)

        return {"analysis": parsed_json}

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in analyze_job:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tailor_resume")
async def tailor_resume(request: TailorResumeRequest):
    try:
        system_prompt = """
You are an expert career coach and technical recruiter with 10+ years experience.
Your job is to tailor resumes to job descriptions and be ATS-friendly.
Always respond with valid JSON only. No extra text.
Do not invent degrees, employers, job titles, certifications, or dates that are not present in the resume text.
"""

        user_prompt = f"""
Tailor the resume to the job description.

Return a JSON object with exactly this structure:
{{
  "match_summary": "1-2 sentences about fit and gaps",
  "missing_keywords": ["keyword1", "keyword2", ...],
  "rewrite_suggestions": [
    {{
      "original": "original bullet from resume (verbatim or close)",
      "rewritten": "improved bullet aligned to the job description"
    }}
  ],
  "skills_section_suggestion": ["skill1", "skill2", ...],
  "warnings": ["warning1", "warning2", ...]
}}

Job Description:
{request.job_description}

Resume Text:
{request.resume_text}
"""

        parsed_json = await _openai_json(system_prompt=system_prompt, user_prompt=user_prompt)
        _apply_defaults(parsed_json, DEFAULT_TAILORED)

        return {"tailored_resume": parsed_json}

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in tailor_resume:", str(e))
        raise HTTPException(status_code=500, detail=str(e))