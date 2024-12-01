import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pathlib
from config import Config
from sklearn.metrics import precision_score
import json


def bool_from_string(b: str) -> bool:
    return True if b == 'True' else False


def get_labels(df):
    map_names = {
        'has asthma': 'asthma',
        ' has COPD': 'copd',
        ' has a cough': 'cough',
        ' has shortness of breath': 'shortness_of_breath',
        ' is a smoker who smokes [insert a realistic number] packs per day':\
            'smoker',
    }

    copd_clinic_list, asthma_education_clinic_list, copd_list, cough_list,\
    shortness_of_breath_list, asthma_list, smoker_list = [], [], [], [], [], [], []


    for i, row in enumerate(df.iter_rows(named=True)):
        labels_dict = {map_names[cond.split(':')[0]]:bool_from_string(cond.split(':')[1])
                       for cond in df['condition'][i].split(',')}
        
        labels_dict['copd_clinic'] = True if labels_dict['copd'] else False
        labels_dict['asthma_education_clinic'] = True\
            if labels_dict['asthma'] else False
        
        copd_clinic_list.append(labels_dict['copd_clinic'])
        asthma_education_clinic_list.append(labels_dict['asthma_education_clinic'])
        copd_list.append(labels_dict['copd'])
        asthma_list.append(labels_dict['asthma'])
        shortness_of_breath_list.append(labels_dict['shortness_of_breath'])
        cough_list.append(labels_dict['cough'])
        smoker_list.append(labels_dict['smoker'])
    
    df = df.with_columns([
        pl.Series("copd_clinic_label", copd_clinic_list),
        pl.Series("asthma_education_clinic_label", asthma_education_clinic_list),
        pl.Series("copd_label", copd_list),
        pl.Series("asthma_label", asthma_list),
        pl.Series("shortness_of_breath_label", shortness_of_breath_list),
        pl.Series("cough_label", cough_list),
        pl.Series("smoker_label", smoker_list)
    ])

    df = df.rename({'other_condition': 'other_label'})

    return df


def get_preds(df):
    copd_clinic_list, asthma_education_clinic_list, copd_list, cough_list,\
    shortness_of_breath_list, asthma_list, smoker_list, other_list =\
        [], [], [], [], [], [], [], []

    for i, row in enumerate(df.iter_rows(named=True)):
        preds = json.loads(df['extraction'][i][8: -4])

        copd_clinic_list.append(preds['copd_clinic'])
        asthma_education_clinic_list.append(preds['asthma_education_clinic'])
        copd_list.append(preds['copd'])
        cough_list.append(preds['cough'])
        shortness_of_breath_list.append(preds['shortness_of_breath'])
        asthma_list.append(preds['asthma'])
        smoker_list.append(preds['smoker'])
        other_list.append(preds['other'])

    df = df.with_columns([
        pl.Series("copd_clinic_pred", copd_clinic_list),
        pl.Series("asthma_education_clinic_pred", asthma_education_clinic_list),
        pl.Series("copd_pred", copd_list),
        pl.Series("asthma_pred", asthma_list),
        pl.Series("shortness_of_breath_pred", shortness_of_breath_list),
        pl.Series("cough_pred", cough_list),
        pl.Series("smoker_pred", smoker_list),
        pl.Series("other_pred", other_list)
    ])

    return df


def calculate_precision_recall_accuracy(df, pred_col, label_col):
    # Calculate true positives and false positives
    tp = df.filter((df[pred_col] == True) & (df[label_col] == True)).height
    fp = df.filter((df[pred_col] == True) & (df[label_col] == False)).height
    tn = df.filter((df[pred_col] == False) & (df[label_col] == False)).height
    fn = df.filter((df[pred_col] == False) & (df[label_col] == True)).height

    # Calculate precision
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0

    # Calculate precision
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0

    # Calculate precision
    accuracy = ((tp + tn) / (tp + fp + tn + fn)) if (tp + fp + tn + fn) > 0 else 1.0
    # Create a new DataFrame with the precision value
    analysis_df = pl.DataFrame({"precision": [precision],
                                 "recall": [recall],
                                 "accuracy": [accuracy],})
    
    return analysis_df


def calculate_text_accuracy(df, pred_col, label_col):
    correct_list = []

    for i, _ in enumerate(df.iter_rows()):
        print(f'\nMANUAL TEXT EVALUATION {i}:')
        print(f'LABEL TEXT:\n{df[label_col][i]}\n')
        print(f'PRED TEXT:\n{df[pred_col][i]}\n')
        label = input(f'ENTER "y" IF MATCHING ELSE "n": ')
        correct_list.append(1 if label == 'y' else 0)
    
    analysis_df = pl.DataFrame(
        {'accuracy': sum(correct_list)/len(correct_list)}
    )

    return analysis_df


def evaluate_single_field(df, field):
    pred_col, label_col = f'{field}_pred', f'{field}_label'
    df = calculate_precision_recall_accuracy(df, pred_col, label_col)
    df = df.rename({
        'precision': f'{field}_precision',
        'recall': f'{field}_recall',
        'accuracy': f'{field}_accuracy',
    })
    return df.select([f'{field}_accuracy'])


def evaluate_multi_field(df, field):
    pred_col, label_col = f'{field}_pred', f'{field}_label'
    df = calculate_precision_recall_accuracy(df, pred_col, label_col)
    df = df.rename({
        'precision': f'{field}_precision',
        'recall': f'{field}_recall',
        'accuracy': f'{field}_accuracy'
    })
    return df


def evaluate_text_field(df, field):
    pred_col, label_col = f'{field}_pred', f'{field}_label'
    df = calculate_text_accuracy(df, pred_col, label_col)
    df = df.rename({
        'accuracy': f'{field}_accuracy'
    })
    return df


def get_html_repr(df):
    # return HTML content
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        h1 {{ text-align: center; }}
        table {{ margin: auto; border-collapse: collapse; width: 80%; }}
        th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Polars DataFrame as Table</h1>
    <table>
        <tr>
            {"".join(f"<th>{col}</th>" for col in df.columns)}
        </tr>
        {"".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in df.rows())}
    </table>
</body>
</html>
"""


def evaluate(extractions_df, fields):
    # dataframe to store all the evaluation results
    eval_df = pl.DataFrame()

    # evaluate all the single fields
    for single_field in fields.single_field:
        new_df = evaluate_single_field(extractions_df, single_field)
        eval_df = pl.concat([eval_df, new_df], how='horizontal') if not eval_df.is_empty() else new_df
    
    # evaluate all the multi fields
    for multi_field in fields.multi_field:
        new_df = evaluate_multi_field(extractions_df, multi_field)
        eval_df = pl.concat([eval_df, new_df], how='horizontal') if not eval_df.is_empty() else new_df
    
    for text_field in fields.text_fields:
        new_df = evaluate_text_field(extractions_df, text_field)
        eval_df = pl.concat([eval_df, new_df], how='horizontal') if not eval_df.is_empty() else new_df

    return eval_df


if __name__ == "__main__":
    """
    This script evaluated the results of the extract_fields script.
    For Single Value Fields, it 
    """
    config = Config()

    # load the extractions dataframe
    extracted_df = pl.read_parquet(config.extraction_output_path)

    extracted_df = get_labels(extracted_df)

    extracted_df = get_preds(extracted_df)

    print(extracted_df.schema)

    # make an evaluations dataframe to store the evaluation results
    eval_df = evaluate(extracted_df, config.fields)

    # save evaluations as dataframe and as html
    pathlib.Path('outputs').mkdir(parents=True, exist_ok=True) 
    eval_df.write_parquet(config.evaluation_output_path)
    html_repr = get_html_repr(eval_df)
    with open(config.evaluation_html_output_path, "w") as file:
        file.write(html_repr)
