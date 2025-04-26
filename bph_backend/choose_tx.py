from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .constants import CHOOSE_TX_LLM

from typing import TypedDict, Union, Literal

class ChooseTxResponse(TypedDict):
    treatments_to_discuss: list[str]

template = """Given the patient's query about BPH - use your knowledge of BPH to select relevant treatments to read about (including e.g. conservative, medical, and surgical options). After your selection, I will provide you with the most updated guidelines for the selected treatments.

If you believe the patient's query warrants information about procedural therapy (e.g. surgery, endoscopic treatment, minimally invasive surgical therapy, etc.), use the following algorithm to choose the precise surgical options:

<surgical considerations>

prostate size:
    'prostate size smaller than 30cc': ['Endoscopic Enucleation (e.g. HoLEP, ThuLEP)', 'Greenlight Photovaporization', 'Transurethral Incision of the Prostate', 'TURP', 'iTIND/Temporarily Implanted Prostate Device'],
    'prostate size between 30-80cc': ['Endoscopic Enucleation (e.g. HoLEP, ThuLEP)', 'Greenlight Photovaporization', 'TURP', 'Urolift/Prostatic Urethral Lift', 'Rezum/Water Vapor Thermal Therapy WVTT', 'Aquablation RWT', 'iTIND/Temporarily Implanted Prostate Device'],
    'prostate size larger than 80cc': ['Simple Prostatectomy (Open, Laparoscopic, Robotic)', 'Endoscopic Enucleation (e.g. HoLEP, ThuLEP)'],

patient values and characteristics:   
    'patient desires sexual preservation (erectile & ejaculatory)': ['Urolift/PUL', 'Rezum/Water Vapor Thermal Therapy WVTT', 'iTIND/Temporarily Implanted Prostate Device', 'Aquablation RWT'],
    'medically comorbid / medically complicated (high risk for general anesthesia; bleeding risk (e.g. taking blood thinners))': ['Endoscopic Enucleation (e.g. HoLEP, ThuLEP)', 'Greenlight Photovaporization']
</surgical considerations>

note that you should always consider all of the factors present in the patient's query (where available) when making your selections; you can mix and match factors to select treatments across the categories.

If the user's query is irrelevant or there are no relevant treatments, respond with an empty list."""

prompt = ChatPromptTemplate([
    ("system", template),
    ("human", "{query}"),
])

choose_tx_chain = prompt | CHOOSE_TX_LLM.with_structured_output(ChooseTxResponse, strict=True, method="json_schema")
