# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
import fastapi.exceptions

# Define the FastAPI app
app = FastAPI()

# Frontend serving logic removed as Vercel will handle it.
# create_frontend_router and app.mount("/app", ...) were here.
