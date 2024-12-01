class Field:
    def __init__(self):
        self.single_field = ['copd_clinic', 'asthma_education_clinic']
        self.multi_field = ['copd', 'asthma', 'shortness_of_breath', 'cough',
                    'smoker']
        self.text_fields = ['other']

class Config:
    def __init__(self):
        self.API_KEY = "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"
        self.generation_output_path = "outputs/data.parquet"
        self.extraction_output_path = "outputs/extractions.parquet"
        self.evaluation_output_path = "outputs/evaluation.parquet"
        self.evaluation_html_output_path = "outputs/evaluation.html"
        self.fields = Field()