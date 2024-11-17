import openai

openai.api_key = '' # Insert API Key

def prompt(condition):
    prompt = f"""
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
9) the reason for the referral (in this case, the patient {condition}).

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
    return prompt

def generate_convo(condition):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a tool that generates patient-doctor conversations with the required specifications."},
            {"role": "user", "content": prompt(condition)}
        ],
        temperature=1.0,
        top_p=0.80,
        max_tokens=4096
    )
    convo = response.choices[0].message.content
    return convo

def save_clinical_notes(condition, text_string, output_filename="clinical_notes.txt"):
    keyword = 'Clinical notes:'
    keyword_index = text_string.find(keyword)

    convo, clinical_notes = "", ""

    if keyword_index != -1:
        convo = text_string[:keyword_index].strip()
        clinical_notes = text_string[keyword_index + len(keyword):].strip()

        with open(output_filename, 'a') as f:
            f.write(f"Condition: {condition}\n")
            f.write(clinical_notes)
            f.write("\n\n")
        print(f"Clinical notes saved to {output_filename}")
    else:
        print("Clinical notes keyword not found in the string.")

    return convo, clinical_notes

conditions = ["has asthma", "has COPD", "has a cough", "has shortness of breath",
              "is a smoker who smokes [insert a realistic number] packs per day"]

with open("conversations.txt", "w") as f:
    pass

with open("clinical_notes.txt", "w") as f:
    pass

with open("conversations.txt", "a") as f:
    for condition in conditions:
        print(f"Condition: {condition}", file=f)
        output = generate_convo(condition)
        convo, notes = save_clinical_notes(condition, output)
        print("Begin Conversation: \n", file=f)
        print(convo, file=f)
        print("\nEnd Conversation", file=f)
        print("\n\n", file=f)

        print(f"Condition: {condition}")
        print("Begin Conversation: \n")
        print(output)
        print("\nEnd Conversation")
        print("\n\n")

