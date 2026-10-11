from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/help", tags=["Help"])

@router.get("/metrics-pdf")
async def metrics_pdf():
    filename = "AItrajectIQ_Metrics_Reference.pdf"

    # Use absolute path relative to this file, so it works regardless of working directory
    pdf_path = Path(__file__).resolve().parents[2] / filename  # app/routes/ -> app/ -> project root
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found on server at {pdf_path}",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={
            # Force download instead of opening inline in the browser
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
