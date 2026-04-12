import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY PREVIEW:", api_key[:16] if api_key else None)
client = OpenAI(api_key=api_key)

app = FastAPI(title="Job Search Assistant")


class JobRequest(BaseModel):
    job_description: str


@app.get("/")
def read_root():
    return {"message": "API running"}


@app.post("/analyze")
def analyze_job(request: JobRequest):
    try:
        prompt = f"""
You are a helpful assistant for job seekers.

Analyze the following job description and return:
1. Role summary
2. Required technical skills
3. Preferred / bonus skills
4. Top 3 resume focus areas

Job Description:
{request.job_description}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for job seekers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        result = response.choices[0].message.content.strip()
        return {"analysis": result}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))