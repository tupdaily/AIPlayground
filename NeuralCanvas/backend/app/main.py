from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NeuralCanvas API",
    description="Backend API for NeuralCanvas — AI-powered visual canvas",
    version="0.1.0",
)

# CORS — allow the Next.js frontend at localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello from NeuralCanvas API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
