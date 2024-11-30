import openai
import polars as pl
import pathlib
from config import Config
import random


def data_gen_prompt(condition, other_condition):
    # return prompts formatted according to the given condition
    system_prompt = "You are a tool that generates patient-doctor conversations with the required specifications."
    additional_info = (
        f" The patient is also experiencing: {other_condition}"
        if other_condition else ""
    )
    user_prompt = f"""
I want you to generate transcripts of conversations between patients and doctors.
The conversations will contain information about a patient being referred to a respiratory specialist.
The Markham Stouffville Hospital will make the referral.
To this end, the conversation should have the following information:
1) whether it is an urgent or a routine referral,
2) the date,
3) the referring physician (referring MD),
4) the MD phone number,
5) the CPSO (College of Physicians and Surgeons of Ontario) number,
6) the preferred language of the patient,
7) the name and number of the interpreter, if the patient needs one (optional),
8) one of either 'COPD Clinic with Respirologist Consultation' or 'Asthma Education Clinic with Asthma Educator (RRT),' and
9) the reason for the referral (in this case, the patient has the following conditions:{condition}).

{additional_info}.

The above is the general referral information.
In terms of patient information, the conversation should contain the following:
1) Hospital MRN (medical record number),
2) Patient name (last name comma first name),
3) Date of birth (in day, month, year format),
4) Sex (female or male),
5) Ontario Health Card number,
6) Version Code,
7) WSIB number (Workplace Safety and Insurance Board),
8) Whether the patient is self-paying (or a refugee),
9) Address with Postal Code,
10) Telephone number (with an optional alternative number), and
11) Email

Ensure that the conversation between the patient and the doctor is not interrogative;
it should be free-flowing, such as a patient who came for a general consultation after which their physician recommends they see a specialist;
or an emergency room doctor urgently ushers their patient to a specialist.
Generate just one conversation.

Additionally, after the conversation, generate doctor's clinical notes for the patient.
These notes could be something like short sentences that give the doctor's thoughts while the talking to the patient, without saying it out loud to the patient.
Ensure that the clinical notes are explicitly identified by "Clinical notes:" to allow for parsing.
"""

    return system_prompt, user_prompt



def gen_convo(condition, other_condition):
    # get prompts for given condition
    system_prompt, user_prompt = data_gen_prompt(condition, other_condition)

    # call OpenAI api with prompt
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1.0,
        top_p=0.80,
        max_tokens=4096
    )
    convo = response.choices[0].message.content

    return convo


def get_convo_clinical_notes(text_string, other_condition):
    # look for Clinical notes keyword in generated output
    keyword = 'Clinical notes:'
    keyword_index = text_string.find(keyword)

    # get the conov and clinical notes from generated output
    convo, clinical_notes = text_string.strip(), ""
    if keyword_index != -1: # split text at clinical notes keyword
        convo = text_string[:keyword_index].strip()
        clinical_notes = text_string[keyword_index + len(keyword):].strip()
        if other_condition:
          clinical_notes += f" Additionally, the patient is experiencing: {other_condition}."
    else:
        print("Clinical notes not found in generated text!")

    return convo, clinical_notes






if __name__ == "__main__":
    config = Config()

    # get the API_KEY
    openai.api_key = config.API_KEY

    # lists to store conditions and the generated conversations
    condition_list = ["has asthma", "has COPD", "has a cough",
                  "has shortness of breath",
                  "is a smoker who smokes [insert a realistic number] packs per day"]

    other_conditions = [
        "Pulmonary Fibrosis", "Interstitial Lung Disease", "Bronchiectasis",
        "Allergic Rhinitis", "Chronic Bronchitis", "Emphysema",
        "Lung Cancer", "Sarcoidosis", "Pneumonia", "Pleural Effusion"
    ]

    convos, clinical_notes, all_conditions, all_other_conditions = [], [], [], []    # Number of examples to generate
    num_examples = 100
    all_examples = []


    for i in range(num_examples):
        print(f"Generating example {i+1} of {num_examples}")
        # Randomly determine how many conditions the patient has
        num_conditions = random.randint(1, len(condition_list))  # Random number of conditions

        # Randomly select the conditions
        selected_conditions = random.sample(condition_list, num_conditions)

        # 20% chance to select a condition, otherwise return None
        other_condition = None
        r = random.random()
        if r <= 0.2:
            other_condition = random.choice(other_conditions)
            print(f"Other condition: {other_condition}")

        # Create a dictionary of all conditions with True/False values
        conditions = {cond: (cond in selected_conditions) for cond in condition_list}
        print(f"Conditions: {conditions}")
        # Convert the conditions to a string format for the prompt
        condition_string = ", ".join(f"{key}:{value}" for key, value in conditions.items())

        output = gen_convo(condition=condition_string, other_condition = other_condition)
        convo, notes = get_convo_clinical_notes(output, other_condition)

        # add convo and clinical notes to list of convos and notes
        convos = convos + [convo]
        clinical_notes = clinical_notes + [notes]
        all_conditions = all_conditions + [condition_string]
        all_other_conditions = all_other_conditions + [other_condition]

    # save generated interactions to df
    df = pl.DataFrame(
        data = {
            "condition": all_conditions,
            "conversation": convos,
            "clinical_note": clinical_notes,
            "other_condition": all_other_conditions
        },
        schema={"condition": pl.String, "conversation": pl.String,
                "clinical_note": pl.String, "other_condition": pl.String}
    )
    print(f"Saving Generated Data to: {config.generation_output_path} \n")
    pathlib.Path('outputs').mkdir(parents=True, exist_ok=True) 
    df.write_parquet(config.generation_output_path)
