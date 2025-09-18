import pandas as pd 
from metrics import get_all_scores
from tqdm import tqdm
import json

with open('config.json', 'r') as f:
    config = json.load(f)

input_path = config["data paths"]["input_path"]

df = pd.read_csv(input_path) 

questions = df['question'].tolist()
answers = df['answer'].tolist()
generated_answers = df['generated'].tolist()

entailment_scores = []
factqa_ps = []
factqa_rs = []
clarity_scores = []
critical_scores = []
pedagogical_scores = []

for i in tqdm(range(len(questions)), desc="Scoring", dynamic_ncols=True):
    question = questions[i]
    answer = answers[i]
    generated = generated_answers[i]
    entailment_score, factqa_p, factqa_r, clarity_score, critical_score, pedagogical_score = get_all_scores(question, answer, generated)
    entailment_scores.append(entailment_score)
    factqa_ps.append(factqa_p)
    factqa_rs.append(factqa_r)
    clarity_scores.append(clarity_score)
    critical_scores.append(critical_score)
    pedagogical_scores.append(pedagogical_score)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'question': questions,
    'answer': answers,
    'generated': generated_answers,
})
if config["metric choices"]["entailment score"]:
    results_df['entailment_score'] = entailment_scores
    print("Average Entailment Score:", sum(entailment_scores) / len(entailment_scores))
if config["metric choices"]["factqa score"]:
    results_df['factqa_precision'] = factqa_ps
    print("Average FactQA Precision:", sum(factqa_ps) / len(factqa_ps))
    results_df['factqa_recall'] = factqa_rs
    print("Average FactQA Recall:", sum(factqa_rs) / len(factqa_rs))
if config["metric choices"]["clarity"]:
    results_df['clarity'] = clarity_scores
    print("Average Clarity Score:", sum(clarity_scores) / len(clarity_scores))
if config["metric choices"]["encouraging critical thinking"]:
    results_df['encouraging critical thinking'] = critical_scores
    print("Average Encouraging Critical Thinking Score:", sum(critical_scores) / len(critical_scores))
if config["metric choices"]["using pedagogical techniques"]:
    results_df['using pedagogical techniques'] = pedagogical_scores
    print("Average Using Pedagogical Techniques Score:", sum(pedagogical_scores) / len(pedagogical_scores))
# Save the results to a CSV file
output_path = config["data paths"]["output_path"] + 'evaluation_results.csv' #change the name of the evaluation results file here
results_df.to_csv(output_path, index=False) #change the name of the evaluation results file here