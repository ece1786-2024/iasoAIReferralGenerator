import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_examples(filePath):
    with open(filePath, "r") as file:
        data = json.load(file)
    actualLabelList = [example["actualLabel"] for example in data["examples"]]
    predictedOutputList = [example["predictedOutput"] for example in data["examples"]]
    return actualLabelList, predictedOutputList

def singleSelectFieldsEvaluation(actualLabel, predictedOutput, fields):
    label = [actualLabel[field] for field in fields]
    predicted = [predictedOutput[field] for field in fields]
    accuracy = accuracy_score(label, predicted)
    return accuracy

def singleCheckboxEvaluationInMultiLabelFields(actualLabel, predictedOutput, field, cumulativeMetrics=None):
    checkboxes = actualLabel[field].keys()
    metrics = {}

    for checkbox in checkboxes:
        label = [actualLabel[field][checkbox]]
        predicted = [predictedOutput[field][checkbox]]

        precision = precision_score(label, predicted, zero_division=0)
        recall = recall_score(label, predicted, zero_division=0)
        accuracy = accuracy_score(label, predicted)

        metrics[checkbox] = {"precision": precision, "recall": recall, "accuracy": accuracy}

        if cumulativeMetrics is not None:
            cumulativeMetrics[checkbox]["precision"].append(precision)
            cumulativeMetrics[checkbox]["recall"].append(recall)
            cumulativeMetrics[checkbox]["accuracy"].append(accuracy)

    averagePrecision = sum([m["precision"] for m in metrics.values()]) / len(metrics)
    averageRecall = sum([m["recall"] for m in metrics.values()]) / len(metrics)
    averageAccuracy = sum([m["accuracy"] for m in metrics.values()]) / len(metrics)

    return metrics, averagePrecision, averageRecall, averageAccuracy

def freeTextFieldsEvaluation(actualLabel, predictedOutput, field):
    actualText = [actualLabel[field].split()]
    predictedText = predictedOutput[field].split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(actualText, predictedText, smoothing_function=smoothing_function)
    return bleu_score

def singleExampleEvaluation(actualLabel, predictedOutput, cumulativeMetrics):
    singleSelectFields = ["COPD_clinic", "Asthma_education"]
    singleSelectFieldsAccuracy = singleSelectFieldsEvaluation(actualLabel, predictedOutput, singleSelectFields)

    multiLabelFields = "clinical_information"
    multiLabelFieldsMetrics, multiLabelPrecision, multiLabelRecall, multiLabelAccuracy = singleCheckboxEvaluationInMultiLabelFields(
        actualLabel, predictedOutput, multiLabelFields, cumulativeMetrics
    )

    freeTextFields = "Other"
    freeTextBleuScore = freeTextFieldsEvaluation(actualLabel, predictedOutput, freeTextFields)

    print("\nMetrics for Multi-Label Fields (Individual Checkboxes):")
    for checkbox, metrics in multiLabelFieldsMetrics.items():
        print(f"\n{checkbox}:\nPrecision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}, Accuracy = {metrics['accuracy']:.2f}")

    print(f"\n\nPer Example Overall Single-Select Fields Accuracy: {singleSelectFieldsAccuracy:.2f}")
    print(f"Per Example Overall Multi-Label Fields Precision: {multiLabelPrecision:.2f}")
    print(f"Per Example Overall Multi-Label Fields Recall: {multiLabelRecall:.2f}")
    print(f"Per Example Overall Multi-Label Fields Accuracy: {multiLabelAccuracy:.2f}")
    print(f"Per Example Overall Free-Text Field BLEU Score: {freeTextBleuScore:.2f}")
    print(f"\n--------------------------------------------------------------------------------")

    return {
        "singleSelectFieldsAccuracy": singleSelectFieldsAccuracy,
        "multiLabelPrecision": multiLabelPrecision,
        "multiLabelRecall": multiLabelRecall,
        "multiLabelAccuracy": multiLabelAccuracy,
        "freeTextBleuScore": freeTextBleuScore,
    }

def multipleExamplesEvaluation(actualLabelList, predictedOutputList):
    totalSingleSelectAccuracy = 0
    totalMultiLabelPrecision = 0
    totalMultiLabelRecall = 0
    totalMultiLabelAccuracy = 0
    totalBleuScore = 0

    cumulativeMetrics = {checkbox: {"precision": [], "recall": [], "accuracy": []}
                          for checkbox in actualLabelList[0]["clinical_information"].keys()}

    n = len(actualLabelList)

    for i, (actualLabel, predictedOutput) in enumerate(zip(actualLabelList, predictedOutputList)):
        print(f"\n------------------------------Evaluating Example {i + 1}------------------------------")
        results = singleExampleEvaluation(actualLabel, predictedOutput, cumulativeMetrics)

        totalSingleSelectAccuracy += results["singleSelectFieldsAccuracy"]
        totalMultiLabelPrecision += results["multiLabelPrecision"]
        totalMultiLabelRecall += results["multiLabelRecall"]
        totalMultiLabelAccuracy += results["multiLabelAccuracy"]
        totalBleuScore += results["freeTextBleuScore"]

    averageSingleSelectAccuracy = totalSingleSelectAccuracy / n
    averageMultiLabelPrecision = totalMultiLabelPrecision / n
    averageMultiLabelRecall = totalMultiLabelRecall / n
    averageMultiLabelAccuracy = totalMultiLabelAccuracy / n
    averageBleuScore = totalBleuScore / n

    print("\nFinal Metrics for Multi-Label Fields (Individual Checkboxes):")
    for checkbox, metrics in cumulativeMetrics.items():
        finalPrecision = sum(metrics["precision"]) / len(metrics["precision"])
        finalRecall = sum(metrics["recall"]) / len(metrics["recall"])
        finalAccuracy = sum(metrics["accuracy"]) / len(metrics["accuracy"])
        print(f"\n{checkbox}:\nPrecision = {finalPrecision:.2f}, Recall = {finalRecall:.2f}, Accuracy = {finalAccuracy:.2f}")

    print("\n\nFinal Evaluation Across All Examples:\n")
    print(f"Average Single-Select Fields Accuracy: {averageSingleSelectAccuracy:.2f}")
    print(f"Average Multi-Label Fields Precision: {averageMultiLabelPrecision:.2f}")
    print(f"Average Multi-Label Fields Recall: {averageMultiLabelRecall:.2f}")
    print(f"Average Multi-Label Fields Accuracy: {averageMultiLabelAccuracy:.2f}")
    print(f"Average Free-Text Field BLEU Score: {averageBleuScore:.2f}")

filePath = "examples.json"
actualLabelList, predictedOutputList = load_examples(filePath)
multipleExamplesEvaluation(actualLabelList, predictedOutputList)