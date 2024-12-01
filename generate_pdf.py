from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import polars as pl
import os
import json
from config import Config

config = Config()
data = pl.read_parquet(config.extraction_output_path)

fieldPositions = {
    "copd_clinic": (66.85, 463),
    "asthma_education_clinic": (66.85, 445),
    "copd": (51, 383.6),
    "asthma": (219, 383.6),
    "cough": (51, 367.75),
    "smoker": (219, 367.75),
    "packs_per_day": (305, 367.75),
    "shortness_of_breath": (51, 352.1),
    "other_checkbox": (51, 334.15),
    "other": (219, 328.15),
}

outputDirectory = "referrals"
os.makedirs(outputDirectory, exist_ok=True)

def parseExtractionFileFields(extraction):
    try:
        cleanExtraction = extraction.strip("`").strip("```json").strip("```")
        parsedData = json.loads(cleanExtraction)
    except json.JSONDecodeError:
        print(f"Error Decoding JSON: {extraction}")
        parsedData = {}

    for field in fieldPositions.keys():
        if field not in parsedData:
            parsedData[field] = None

    return parsedData

def annotate(unannotatedPdfPath, annotatedPdfPath, extractedFields):
    reader = PdfReader(unannotatedPdfPath)
    writer = PdfWriter()
    unannotatedFirstPage = reader.pages[0]
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)

    for field, position in fieldPositions.items():
        value = extractedFields.get(field)
        if field in ["copd_clinic", "asthma_education_clinic", "copd", "asthma", "shortness_of_breath", "cough", "smoker"]:
            if value:
                can.drawString(position[0], position[1], "✔")
        elif field == "packs_per_day" and value is not None and value > 0:
            can.drawString(position[0], position[1], str(value))
        elif field == "other" and value:
            can.drawString(fieldPositions["other_checkbox"][0], fieldPositions["other_checkbox"][1], "✔")
            can.drawString(position[0], position[1], value)

    can.showPage()
    can.save()
    packet.seek(0)

    overlay_pdf = PdfReader(packet)
    if len(overlay_pdf.pages) > 0:
        unannotatedFirstPage.merge_page(overlay_pdf.pages[0])
    else:
        print("No Annotations Added.")
        return

    writer.add_page(unannotatedFirstPage)

    for pageNumber in range(1, len(reader.pages)):
        writer.add_page(reader.pages[pageNumber])

    with open(annotatedPdfPath, "wb") as out_pdf:
        writer.write(out_pdf)

unannotatedPdfPath = "M-CRHR-9-18.pdf"
for i, row in enumerate(data.to_dicts()):
    extractedFields = parseExtractionFileFields(row["extraction"])
    annotatedPdfPath = f"{outputDirectory}/Referral Form {i + 1}.pdf"
    annotate(unannotatedPdfPath, annotatedPdfPath, extractedFields)
    print(f"Generated: {annotatedPdfPath}")