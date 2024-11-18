# iasoAIReferralGenerator

An agentic system to automatically populate a doctor's referral form
that allows a patient to access specialized care. This system is a
proof-of-concept which generates a doctor's referral form based on 
patient-doctor conversation transcript and a doctor's clinical note.

## Usage
Instructions here

``` bash
bash instructions here
```

## Format

This system populates this referral form in question from 
Markham-Stouffeville Hospital for their Centre of Respiratory Health
Referral. 

Link: https://www.oakvalleyhealth.ca/wp-content/uploads/2022/09/M-CRHR-9-18.pdf

This form contains fields that will either be autopopulated by a doctor's 
computer system such as patient information like Name, Address, Health Card
Number. Some other fields would be filled in by the Hospital Staff and thus 
are also not populated by this system. The fields that are autopopulated by
this system are:

```
{
    "copd_clinic": True, # boolean single field
    
    "asthma_education": False, # boolean single field
    
    "clinical_information": {
    
        "copd": False, # boolean multi field
    
        "cough": True, # boolean multi field
    
        "shortness_of_breath": True, # boolean multi field
    
        "asthma": False, # boolean multi field
    
        "smoking": False, # boolean multi field
    
        "smoking_packs_per_day": 0, # integer multi field
    
    },
    
    "other": "Doctor's notes will go here"
}
```
