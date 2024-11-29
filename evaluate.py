import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pathlib
from config import Config


def evaluate_single_field(df, field):
    df = df.with_columns(
        (pl.col(f'{field}_pred') == pl.col(f'{field}_label'))
        .alias(f'{field}_accuracy')
    )
    df = df[f'{field}_accuracy'].mean()
    return df


def evaluate_multi_field(df, field):
    df = df.with_columns_seq(
        (pl.col(f'{field}_pred') and pl.col(f'{field}_label'))
        .alias(f'{field}_tp'),
        (pl.col(f'{field}_pred') and not pl.col(f'{field}_label'))
        .alias(f'{field}_fp'),
        (not pl.col(f'{field}_pred') and not pl.col(f'{field}_label'))
        .alias(f'{field}_tn'),
        (not pl.col(f'{field}_pred') and pl.col(f'{field}_label'))
        .alias(f'{field}_fn'),
        (pl.col(f'{field}_tp') or pl.col(f'{field}_tn'))
        .alias(f'{field}_tp_or_tn'),
        (pl.col(f'{field}_tp') or pl.col(f'{field}_fp'))
        .alias(f'{field}_tp_or_fp'),
        (pl.col(f'{field}_tp') or pl.col(f'{field}_fn'))
        .alias(f'{field}_tp_or_fn')
    )
    df = df[f'{field}_accuracy'].mean()
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
    
    return eval_df


if __name__ == "__main__":
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
