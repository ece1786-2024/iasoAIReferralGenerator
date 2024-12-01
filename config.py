class Field:
    def __init__(self):
        self.single_field = ['copd_clinic', 'asthma_education_clinic']
        self.multi_field = ['copd', 'asthma', 'shortness_of_breath', 'cough',
                    'smoker', 'packs_per_day']
        self.text_fields = ['other']

class Config:
    def __init__(self):
        self.API_KEY = "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"
        self.generation_output_path = "data/data.parquet"
        self.extraction_output_path = "outputs/extractions.parquet"
        self.evaluation_output_path = "outputs/evaluation.parquet"
        self.evaluation_html_output_path = "outputs/evaluation.html"
        self.fields = Field()