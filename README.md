# Job Search API

## Overview
This project is a backend API that automates job search and application workflows.  
It allows users to fetch job listings, process them, and streamline application-related tasks.

The system is built with FastAPI and containerized using Docker for easy deployment.

## How to Run

### Run locally
```
pip install -r requirements.txt
uvicorn app.main:app --reload

```

### Run with Docker

```
docker build -t job-api .
docker run -p 8000:8000 job-api
```
