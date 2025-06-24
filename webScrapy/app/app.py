# app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import subprocess
import uuid
import os
import pandas as pd


app = FastAPI()

@app.get("/scrape")
def scrape( format: str = "json"):
    run_id = str(uuid.uuid4())
    output_file = f"output_{run_id}.{format}"

    spider_dir = os.path.abspath("../table_extractor")  # Absolute path to your Scrapy project

    cmd = [
        "scrapy", "crawl", "table_spider",
        "-o", output_file,
    ]

    subprocess.run(cmd, check=True, cwd=spider_dir)  # <== Working directory is set here

    output_path = os.path.join(spider_dir, output_file)

    if format == "json":
        data = pd.read_json(output_path)
        os.remove(output_path)
        print(data)
        return JSONResponse(content=data.to_dict(orient="records"))
    elif format == "csv":
        return FileResponse(output_path, filename=output_file, media_type="text/csv")
