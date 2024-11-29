class Field:
    def __init__(self):
        self.single_field = ['copd_clinic', 'asthma_education_clinic']
        self.multi_field = ['copd', 'asthma', 'shortness_of_breath', 'cough',
                    'smoker', 'packs_per_day']
        self.text_fields = ['other']

class Config:
    def __init__(self):
        self.API_KEY = "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"
        self.generation_output_path = "outputs/data.parquet"
        self.extraction_output_path = "outputs/extractions.parquet"
        self.evaluation_output_path = "outputs/evaluation.parquet"
        self.evaluation_html_output_path = "outputs/evaluation.html"
        self.fields = Field()