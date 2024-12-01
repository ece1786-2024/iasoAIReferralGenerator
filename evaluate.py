import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pathlib
from config import Config
from sklearn.metrics import precision_score


def calculate_precision_recall_accuracy(df, pred_col, label_col):
    # Calculate true positives and false positives
    tp = ((df[pred_col] == 1) & (df[label_col] == 1)).sum()
    fp = ((df[pred_col] == 1) & (df[label_col] == 0)).sum()
    tn = ((df[pred_col] == 0) & (df[label_col] == 0)).sum()
    fn = ((df[pred_col] == 0) & (df[label_col] == 1)).sum()
    
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    # Calculate precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    # Calculate precision
    accuracy = tp + tn / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 1.0
    
    # Create a new DataFrame with the precision value
    analysis_df = pl.DataFrame({"precision": [precision],
                                 "recall": [recall],
                                 "accuracy": [accuracy],})
    
    return analysis_df


def evaluate_single_field(df, field):
    pred_col, label_col = f'{field}_pred', f'{field}_pred'
    df = calculate_precision_recall_accuracy(df, pred_col, label_col)
    df = df.rename({
        'precision': f'{field}_precision',
        'recall': f'{field}_recall',
        'accuracy': f'{field}_accuracy',
    })
    return df[f'{field}_accuracy']


def evaluate_multi_field(df, field):
    pred_col, label_col = f'{field}_pred', f'{field}_pred'
    df = calculate_precision_recall_accuracy(df, pred_col, label_col)
    df = df.rename({
        'precision': f'{field}_precision',
        'recall': f'{field}_recall',
        'accuracy': f'{field}_accuracy'
    })
    return df


def evaluate_text_field(df, field):
    pass


def get_html_repr(df):
    # return HTML content
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Polars DataFrame Table</title>
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
    for single_field in fields['single_fields']:
        new_df = evaluate_single_field(extractions_df, single_field)
        eval_df = pl.concat([eval_df, new_df]) if not eval_df.is_empty() else new_df
    
    # evaluate all the multi fields
    for multi_field in fields['multi_fields']:
        new_df = evaluate_multi_field(extractions_df, multi_field)
        eval_df = pl.concat([eval_df, new_df]) if not eval_df.is_empty() else new_df
    
    return eval_df


if __name__ == "__main__":
    """
    This script evaluated the results of the extract_fields script.
    For Single Value Fields, it 
    """
    config = Config()

    # load the extractions dataframe
    extracted_df = pl.load_parquet(config.extraction_output_path)

    # make an evaluations dataframe to store the evaluation results
    eval_df = evaluate(extracted_df, config.fields)

    # save evaluations as dataframe and as html
    pathlib.Path('outputs').mkdir(parents=True, exist_ok=True) 
    eval_df.write_parquet(config.generation_output_path)
    html_repr = get_html_repr(eval_df)
    with open(config.evaluation_html_output_path, "w") as file:
        file.write(html_repr)
