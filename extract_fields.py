import openai
import polars as pl
import pathlib
from config import Config


def get_extraction_prompts(conversation, clinical_notes):
    # return prompts formatted according to the given data
    system_prompt = """
You are an assistant designed to extract and organize information from patient-doctor conversations and clinical notes to fill out a "Centre for Respiratory Health Referral Form." 

The output should be in JSON format and must include the following fields. For any checkbox fields, output `true` if checked or `false` if not. For fields with predefined values, ensure the values match the specified options.

---

**JSON Structure and Field Details**

  - `"copd_clinic"`: `true` or `false`.
  - `"asthma_education_clinic"`: `true` or `false`.
  - `"copd"`: `true` or `false`.
  - `"asthma"`: `true` or `false`.
  - `"shortness_of_breath"`: `true` or `false`.
  - `"cough"`: `true` or `false`.
  - `"asthma"`: `true` or `false`.
  - `"smoker"`: `true` or `false`.
  - `"packs_per_day"`: Number of packs smoked per day (integer or 0 if non-smoker).
  - `"other"`: String description of other reasons (string or "" empty string if unavailable).

---

**Additional Notes:**
1. For missing or unavailable information, use false or 0 or "" respectively.
2. All fields must be included in the JSON even if they are false, 0, or "".
3. For checkboxes, explicitly state `true` for checked and `false` for unchecked.

---

Use the patient-doctor conversation and clinical notes to populate these fields. If any field cannot be filled directly from the provided data, infer details when reasonable, and leave as false, 0, or "" as appropriate if inference is not possible. Output only the JSON.

"""

    user_prompt = f"""
Here are the patient-doctor conversation and clinical notes:

**Patient-Doctor Conversation:**
{conversation}

**Clinical Notes:**
{clinical_notes}

Using the provided conversation, clinical notes, and extractions, populate the "Centre for Respiratory Health Referral Form" fields as described in the instructions. Ensure that:
- All checkbox fields are `true` or `false`.
- Missing or unavailable fields are set to false for binary fields, 0 for integer fields, and "" for text fields.

Output the result as a JSON object.

"""

    return system_prompt, user_prompt


def get_verification_prompts(conversation, clinical_notes, extract1,
                             extract2, extract3):
    system_prompt = """
You are an assistant designed to verify extracted fields from patient-doctor conversations and clinical notes to fill out a "Centre for Respiratory Health Referral Form." 

The extracted fields are in the form of a json an may contain conflicted values. Given the conversation and clinical notes complete another json.

The output should be in JSON format and must include the following fields. For any checkbox fields, output `true` if checked or `false` if not. For fields with predefined values, ensure the values match the specified options.

---

**JSON Structure and Field Details**

  - `"copd_clinic"`: `true` or `false`.
  - `"asthma_education_clinic"`: `true` or `false`.
  - `"copd"`: `true` or `false`.
  - `"asthma"`: `true` or `false`.
  - `"shortness_of_breath"`: `true` or `false`.
  - `"cough"`: `true` or `false`.
  - `"smoker"`: `true` or `false`.
  - `"packs_per_day"`: Number of packs smoked per day (integer or 0 if non-smoker).
  - `"other"`: String description of other reasons (string or "" empty string if unavailable).

---

**Additional Notes:**
1. For missing or unavailable information, use false or 0 or "" respectively.
2. All fields must be included in the JSON even if they are false, 0, or "".
3. For checkboxes, explicitly state `true` for checked and `false` for unchecked.

---

Use the patient-doctor conversation and clinical notes to populate these fields. If any field cannot be filled directly from the provided data, infer details when reasonable, and leave as false, 0, or "" as appropriate if inference is not possible. Output only the JSON.

"""

    user_prompt = f"""
Here are the patient-doctor conversation, clinical notes and the extracted JSONs:

**Patient-Doctor Conversation:**
{conversation}

**Clinical Notes:**
{clinical_notes}

**Extraction1**
{extract1}

**Extraction2**
{extract2}

**Extraction3**
{extract3}

Using the provided conversation, clinical notes, and extractions, populate the "Centre for Respiratory Health Referral Form" fields as described in the instructions. Ensure that:
- All checkbox fields are `true` or `false`.
- Missing or unavailable fields are set to false for binary fields, 0 for integer fields, and "" for text fields.

Output the result as a JSON object.

"""

    return system_prompt, user_prompt


def extract_fields(conversation: str, clinical_notes: str) -> str:
    # get prompts
    system_prompt, user_prompt = get_extraction_prompts(conversation,
                                                        clinical_notes)

    # get api response
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.2,
        top_p=1.0
    )
    extracted_info = response.choices[0].message.content.strip()
    
    return extracted_info


def verify_fields(conversation: str, clinical_notes: str,
                  extraction1: str, extraction2: str, extraction3: str) -> str:
    # get prompts
    system_prompt, user_prompt = get_verification_prompts(conversation,
                                                        clinical_notes,
                                                        extraction1,
                                                        extraction2,
                                                        extraction3)

    # get api response
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.0,
        top_p=1.0
    )
    verified_extraction = response.choices[0].message.content.strip()
    
    return verified_extraction


def extract_and_verify_fields(conversation: str, clinical_note: str) -> str:
    # generate 3 different extractions for each set of
    # conversations and clinical notes
    extraction1 = extract_fields(conversation, clinical_note)
    extraction2 = extract_fields(conversation, clinical_note)
    extraction3 = extract_fields(conversation, clinical_note)

    # generate a verified extraction using the initial extraction
    verified_extraction = verify_fields(conversation, clinical_note,
                                        extraction1, extraction2,
                                        extraction3)

    return verified_extraction


if __name__ == "__main__":
    config = Config()

    # get the API_KEY
    openai.api_key = config.API_KEY

    # load generated data parquet and call extraction on each row
    extractions = []
    data_df = pl.read_parquet(config.generation_output_path)
    print("Extracting Relevant Information from Conversation and Clinical Notes:")
    for i, (conversation, clinical_note) in enumerate(zip(data_df["conversation"],
                                           data_df["clinical_note"])):
        print(f"Extracting Details for Form {i}.")
        extraction = extract_and_verify_fields(conversation, clinical_note)
        extractions = extractions + [extraction]

    # add extractions columns to dataframe and save it out to parquet
    extractions = pl.Series("extraction", extractions)
    extracted_df = data_df.insert_column(3, extractions)
    print(f'Saving Extracted Data to: {config.extraction_output_path}\n')
    pathlib.Path('outputs').mkdir(parents=True, exist_ok=True) 
    extracted_df.write_parquet(config.extraction_output_path)
