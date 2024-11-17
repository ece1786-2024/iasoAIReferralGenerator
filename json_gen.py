import openai

openai.api_key = ""

# Define the function to process conversation
def extract_information_with_gpt(conversation: str, clinical_notes: str) -> str:
    system_prompt = """
You are an assistant that extracts structured information from patient-doctor conversations and clinical notes to fill out medical forms. 
The forms have the following fields:

**Form 1: Echocardiogram Requisition**
- Patient’s Name
- Date of Birth
- Health Card Number
- Telephone
- Additional Reports To
- Referring Doctor’s Name
- Indications (list of checked conditions)
- Relevant Clinical Information
- Echocardiogram Type

**Form 2: Acute Respiratory Clinic Referral**
- Last Name
- First Name
- Date of Birth
- Health Card Number
- Address
- Emergency Contact Name
- Emergency Contact Phone Number
- Referring Physician
- Clinical Information/Reason for Referral
- Diagnostic Requirements (Renal insufficiency, etc.)

Use both the provided patient-doctor conversation and clinical notes to extract relevant information. 
Where information is missing or ambiguous, infer details based on the context of the conversation and clinical notes. If nothing is possible, set it to null.

Structure the output as a JSON object with separate dictionaries for each form.
"""

    user_prompt = f"""
    Here are the patient-doctor conversation and clinical notes:

    **Patient-Doctor Conversation:**
    {conversation}

    **Clinical Notes:**
    {clinical_notes}

    Please extract the information and organize it as a JSON object, separating the fields for both forms. Ensure each form's fields are completed based on the provided information.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(statement=conversation, clinical_notes = clinical_notes)}
        ],
        max_tokens=500,
        temperature=0,
        top_p=1.0
    )
    extracted_info = response.choices[0].message.content.strip()
    
    return extracted_info

# Testing (Uncomment the following lines to test the function)


# conversation = """
# Patient: I've been experiencing chest pain and shortness of breath for the past week.
# Doctor: Have you ever had similar symptoms before?
# Patient: Yes, I had some chest pain a few years ago, but it wasn't this bad.
# Doctor: Any history of heart disease or hypertension?
# Patient: I was diagnosed with hypertension two years ago.
# Doctor: Are you currently taking any medication for hypertension?
# Patient: Yes, I am on amlodipine.
# Doctor: I’ll recommend an echocardiogram to investigate further and will refer you to the Acute Respiratory Clinic for your shortness of breath.
# """

# clinical_notes = """
# - Patient Name: John Doe
# - Date of Birth: 1980-01-15
# - Health Card Number: 1234-5678-90
# - Address: 456 Elm Street, Toronto, ON, M1A 2B3
# - Contact: Home - (123) 456-7890, Emergency - Jane Doe (Spouse), (987) 654-3210
# - Known Conditions: Hypertension, suspected structural heart disease
# - Medications: Amlodipine (5 mg daily)
# - Reason for Referral: Persistent shortness of breath and chest pain, ruling out coronary artery disease and pulmonary issues.
# - Diagnostic Requirements: Needs echocardiogram and respiratory consultation. Bloodwork: Normal, recent CT not yet arranged.
# """

# extracted_info = extract_information_with_gpt(conversation, clinical_notes)
# print(extracted_info)