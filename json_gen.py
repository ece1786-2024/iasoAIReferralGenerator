import openai

openai.api_key = ""

def extract_information_with_gpt(conversation: str, clinical_notes: str) -> str:
    system_prompt = """
You are an assistant designed to extract and organize information from patient-doctor conversations and clinical notes to fill out a "Centre for Respiratory Health Referral Form." 

The output should be in JSON format and must include the following fields. For any checkbox fields, output `true` if checked or `false` if not. For fields with predefined values, ensure the values match the specified options.

---

**JSON Structure and Field Details**

1. **General Information**:
   - `"date"`: Referral date in `DD/MM/YYYY` format.
   - `"referring_md"`:
     - `"name"`: Name of the referring doctor (string).
     - `"phone_number"`: Contact number of the referring doctor (string).
     - `"cpso_number"`: CPSO number of the referring doctor (string or `null` if unavailable).
   - `"additional_copies_to"`: List of names to send additional copies to (array of strings or an empty array if none).

2. **Patient Information**:
   - `"patient_name"`: Full name in `"Last, First"` format (string).
   - `"date_of_birth"`: Patient's date of birth in `DD/MM/YYYY` format.
   - `"sex"`: `"F"` for Female, `"M"` for Male, or `"Other"` if unspecified.
   - `"health_card_number"`: Patient’s health card number (string).
   - `"version_code"`: Health card version code (string or `null` if unavailable).
   - `"wsib_number"`: WSIB number (string or `null` if unavailable).
   - `"non_ohip_status"`: `"Self-pay"`, `"Refugee"`, or `"None"` if not applicable.
   - `"address"`: Full patient address (string).
   - `"postal_code"`: Postal code of the address (string).
   - `"contact_information"`:
     - `"best_daytime_phone"`: Best daytime phone number (string).
     - `"alternate_phone"`: Alternate phone number (string or `null` if unavailable).
     - `"email"`: Patient email address (string or `null` if unavailable).
   - `"preferred_language"`: Patient’s preferred language (string or `null` if unavailable).

3. **Referral Details**:
   - `"reason_for_referral"`:
     - `"copd"`: `true` or `false`.
     - `"asthma"`: `true` or `false`.
     - `"shortness_of_breath"`: `true` or `false`.
     - `"cough"`: `true` or `false`.
     - `"other"`: String description of other reasons (string or `null` if unavailable).
   - `"clinical_information"`: Detailed clinical notes relevant to the referral (string).

4. **Clinic Preferences**:
   - `"clinic_type"`:
     - `"copd_clinic"`: `true` or `false`.
     - `"asthma_education_clinic"`: `true` or `false`.
   - `"urgency"`: `"Urgent"` or `"Routine"`.

5. **Appointments and Tests**:
   - `"chest_xray_required"`: `true` or `false`.
   - `"pft_required"`: `true` or `false`.
   - `"clinic_appointment"`:
     - `"date"`: Date of the appointment in `DD/MM/YYYY` format or `null` if unavailable.
     - `"time"`: Time of the appointment in `HH:MM` format or `null` if unavailable.
   - `"rrt_information"`:
     - `"name"`: Name of the RRT (string or `null` if unavailable).
     - `"signature"`: Signature of the RRT (string or `null` if unavailable).

6. **Smoking History**:
   - `"smoker"`: `true` or `false`.
   - `"packs_per_day"`: Number of packs smoked per day (integer or `null` if non-smoker).

7. **Interpreter Information**:
   - `"interpreter"`:
     - `"name"`: Name of the interpreter (string or `null` if unavailable).
     - `"contact_number"`: Contact number of the interpreter (string or `null` if unavailable).

---

**Additional Notes:**
1. For missing or unavailable information, use `null`.
2. All fields must be included in the JSON even if they are `null`.
3. For checkboxes, explicitly state `true` for checked and `false` for unchecked.
4. For time-related fields, use a 24-hour format (e.g., `14:30` for 2:30 PM).

---

Use the patient-doctor conversation and clinical notes to populate these fields. If any field cannot be filled directly from the provided data, infer details when reasonable, and leave as `null` if inference is not possible. Output only the JSON.

"""

    user_prompt = f"""
    Here are the patient-doctor conversation and clinical notes:

**Patient-Doctor Conversation:**
{conversation}

**Clinical Notes:**
{clinical_notes}

Using the provided conversation and clinical notes, populate the "Centre for Respiratory Health Referral Form" fields as described in the instructions. Ensure that:
- All checkbox fields are `true` or `false`.
- Missing or unavailable fields are set to `null`.
- Dates and times are in the specified formats.

Output the result as a JSON object.

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