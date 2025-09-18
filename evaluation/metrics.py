from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import torch
from llm_inference import gpt4_response, gemini_response
import re
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# Load the Entailment model
ent_model = CrossEncoder('cross-encoder/nli-roberta-base', device='cpu') #change to 'cuda' if you have a GPU available

def entailment(question, answer, generated):
    sentence_pairs = (generated, answer)
    scores = ent_model.predict(sentence_pairs)
    # Convert logits to probabilities using softmax
    #print(scores)
    probabilities = []
    probs = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)
    #print(probs)
    probabilities = probs.tolist()
    #print(probabilities)
    # label_mapping = ['contradiction', 'entailment', 'neutral']
    return probabilities[1]

def factqa_prompt(question: str, answer_1: str, answer_2: str):
    question = question.replace("\n", " ").strip()
    answer_1 = answer_1.replace("\n", " ").strip()
    answer_2 = answer_2.replace("\n", " ").strip()
    return "Your job is to evaluate the similarity of different answers to a single question. " +\
        "You will be given a question from a specific computer science college course. " +\
        "You will also be given two possible answers to that question, " +\
        "and will have to evaluate the claims in one answer against the other.\n\n" +\
        "Steps:\n" +\
        "1. List all of the atomic claims made by Answer 1. " +\
        "Note that an answer saying that there is no information counts as a single claim.\n" +\
        "2. Tell me which of those claims are supported by Answer 2.\n" +\
        "3. Summarize the results using the template \"Score: <num supported claims>/<num total claims>\". " +\
        "Ensure that both numbers are integers.\n\n" +\
        f"Question: {question}\n" +\
        f"Answer 1: {answer_1}\n" +\
        f"Answer 2: {answer_2}"

def extract_score(input_string):
    # Regular expression to find the score pattern "Score: x/y"
    score_match = re.search(r"Score: (\d+/\d+)", input_string)
    if score_match:
        # Extracting the score string
        score_str = score_match.group(1)
        # Splitting to get numerator and denominator
        numerator, denominator = map(int, score_str.split('/'))
        # Converting to float by dividing numerator by denominator
        return numerator / denominator
    else:
        # Return None if no score pattern is found
        return None

def extract_json(s):
    try:
        # Attempt to decode the entire string as JSON
        obj = json.loads(s)
        return json.dumps(obj, indent=2)  # Return a nicely formatted JSON string
    except json.JSONDecodeError:
        # If the above fails, attempt to locate JSON by finding balanced curly braces
        stack = []
        for i, char in enumerate(s):
            if char == '{':
                stack.append(i)
            elif char == '}':
                start = stack.pop()
                if not stack:  # When the stack is empty, we've closed the outermost brace
                    try:
                        # Try to parse the substring as JSON
                        obj = json.loads(s[start:i+1])
                        return json.dumps(obj, indent=2)  # Return a nicely formatted JSON string
                    except json.JSONDecodeError:
                        continue  # Continue if substring is not valid JSON
    return None


def factqa(question, answer, generated):
    factqa_precision_prompt = factqa_prompt(question, answer, generated)
    factqa_recall_prompt = factqa_prompt(question, generated, answer)
    factqa_precision = gpt4_response(factqa_precision_prompt) #can change to gemini_response if needed
    factqa_recall = gpt4_response(factqa_recall_prompt) #can change to gemini_response if needed
    precision_score = extract_score(factqa_precision) 
    recall_score = extract_score(factqa_recall)
    if precision_score is None or recall_score is None:
        return 0.0, 0.0
    else:
        return precision_score, recall_score #returns factqa precision and recall scores as a tuple

def clarity(question, answer, generated): #score for 'Clarity'
    SYSTEM_PROMPT = "You are a domain expert in Computer Science. You are given a question and an answer. Judge the answer and give an appropriate score."
    TASK_INSTRUCTION = '''
    Task Instructions: You are a strict grader. Judge the answer on the following criterion:  
    • Clarity (simplifies complex terms, logically structured, unambiguous)  

    Rules:
    Count unexplained jargon terms → Count transition phrases (e.g., "so", "therefore", "next") → Detect ambiguous or compound run-on statements.

    Assign a score based on the following scale:

    1 = >=2 jargon terms without explanation, and >=2 incoherent transitions.  
    2 = >= 1 jargon term unexplained and at least 1 logical jump or ambiguous phrasing. 
    3 = Mostly clear, but 1–2 minor issues: one ambiguous phrase or slightly choppy flow. 
    4 = All terms explained, clear flow, no ambiguity except possibly 1 unclear phrase.
    5 = No unexplained jargon, consistent logical flow, zero ambiguity.
    '''

    OUTPUT_FORMAT = '''
    Output Format: Return **only** the JSON object below—no extra text.

    ```json
    {
    "overall": <1-5>
    }
    '''

    examples = '''
    Here are some examples:
    Example 1:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: BSTs optimise O(log n) retrieval; apex node bifurcates sub-trees ergo bigger left subchild contrarily right.
    Output: {
        "explanation": "Two jargon terms (O(log n), apex), two incoherent jumps (“ergo”, “contrarily”): fails both thresholds.",
        "overall": 1
        }
    Example 2:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: A BST has nodes and children. Therefore, data is stored efficiently.
    Output: {
        "explanation": "One unexplained term (nodes), one logical jump (why does it imply efficiency?)",
        "overall": 2
        }
    Example 3:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: A BST is a tree where each left child holds a smaller value than its parent, and the right child a larger value. This rule lets us skip half the tree each step, making searches fast.
    Output: {
        "explanation": "Mostly clear, but phrase “skip half the tree” is mildly ambiguous about how.",
        "overall": 3
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: A Binary Search Tree (BST) is an ordered tree: every node’s left subtree contains only smaller keys, the right subtree only larger keys. Following this rule top-down lets you discard half the remaining elements each comparison.
    Output: {
        "explanation": "All terms defined; flow is clear; one sentence is long but unambiguous.",
        "overall": 4
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: A Binary Search Tree (BST) is a sorted tree structure. For each node: left < parent < right. Starting at the root you compare the target key: if it is smaller, move left; if larger, move right. Repeating this till you reach a leaf takes at most log₂ n steps. No jargon remains unexplained, and each step follows directly from the previous one.
    Output: {
        "explanation": "Zero jargon, crisp stepwise flow, no ambiguity.",
        "overall": 5
        }
        
    '''

    QUERY_PROMPT = "Given the following question and answers, assign the appropriate score and give the explanation as shown in the examples. Output the dictionary format as described above. Do not include any other text. "
    final_prompt = SYSTEM_PROMPT + TASK_INSTRUCTION + OUTPUT_FORMAT + examples + QUERY_PROMPT + "[Q]: " + question + "Answer: " + generated + " Output: " #change

    response = gpt4_response(final_prompt) #can change to gemini_response if needed
    #print(response)
    # Extract the overall score from the response
    extracted_json = extract_json(response)
    score = json.loads(extracted_json)["overall"]
    return score

def critical(question, answer, generated): #score for 'Encouraging Critical Thinking'
    SYSTEM_PROMPT = "You are a domain expert in Computer Science. You are given a question and an answer. Judge the answer and give an appropriate score."
    TASK_INSTRUCTION = '''
    Task Instructions: You are a strict grader. Judge the answer on the following criterion:  
    • Encouraging Critical Thinking (Prompts learners to reflect, explore alternatives, or ask new questions.)  

    Rules:
    Detect open-ended question marks ("why", "how", "what if"), alternatives (“another way”, “alternatively”, “one approach is…”), and exploration prompts (“you may explore”, “consider trying…”).

    Assign a score based on the following scale:

    1 = No questions, no alternatives, purely factual.  
    2 = Includes 1 suggestive or reflective phrase, but no actual open-ended question. 
    3 = Contains 1 open-ended question or 1 alternative method/viewpoint. 
    4 = >=2 open-ended prompts or multiple viewpoints briefly compared.
    5 = >=2 open-ended questions + explicit invitation to explore further.
    '''

    OUTPUT_FORMAT = '''
    Output Format: Return **only** the JSON object below—no extra text.

    ```json
    {
    "explanation": "<explanation>",
    "overall": <1-5>
    }
    '''

    examples = '''
    Here are some examples:
    Example 1:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: BSTs are efficient search structures used in programming.
    Output: {
        "explanation": "Purely factual; no questions or alternatives.",
        "overall": 1
        }
    Example 2:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: BSTs are efficient; it is worth thinking about their balance.
    Output: {
        "explanation": "Reflective phrase “worth thinking” but no open-ended question.",
        "overall": 2
        }
    Example 3:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: How would search time change if the tree became unbalanced?
    Output: {
        "explanation": "One open-ended question prompts reflection.",
        "overall": 3
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: What happens if the tree degenerates into a list—and can you think of another structure that avoids this? Compare that with self‑balancing trees such as AVL.
    Output: {
        "explanation": "Two prompts: a “what happens” question plus an alternative to explore.",
        "overall": 4
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: Why might a hash table outperform a BST for large data sets? After trying a BST yourself, consider experimenting with AVL or Red‑Black trees and evaluate which conditions favour each structure.
    Output: {
        "explanation": ">=2 open questions and an explicit invitation to explore further.",
        "overall": 5
        }
        
    '''

    QUERY_PROMPT = "Given the following question and answers, assign the appropriate score and give the explanation as shown in the examples. Output the dictionary format as described above. Do not include any other text. "
    final_prompt = SYSTEM_PROMPT + TASK_INSTRUCTION + OUTPUT_FORMAT + examples + QUERY_PROMPT + "[Q]: " + question + "Answer: " + generated + " Output: " #change
    response = gpt4_response(final_prompt) #can change to gemini_response if needed
    #print(response)
    # Extract the overall score from the response
    extracted_json = extract_json(response)
    score = json.loads(extracted_json)["overall"]
    return score

def pedagogical(question, answer, generated): #score for 'Using Pedagogical Techniques'
    SYSTEM_PROMPT = "You are a domain expert in Computer Science. You are given a question and an answer. Judge the answer and give an appropriate score."
    TASK_INSTRUCTION = '''
    Task Instructions: You are a strict grader. Judge the answer on the following criterion:  
    • Using Pedagogical Techniques (Employs examples, analogies, or step-by-step explanations to aid understanding)  

    Rules:
    Search for example phrases (“for example”, “e.g.”), analogies (“like”, “similar to”), step phrases (“Step 1”, “First,” “Then”).

    Assign a score based on the following scale:

    1 = Pure explanation without any example or breakdown.  
    2 = 1 brief example or partial list of steps, lacking clarity. 
    3 = 1 complete example or full step list present, but not both. 
    4 = >=2 teaching techniques used (e.g., example + step list), with moderate clarity.
    5 = >=3 techniques (e.g., example, analogy, visual mention), all clear and complete.
    '''

    OUTPUT_FORMAT = '''
    Output Format: Return **only** the JSON object below—no extra text.

    ```json
    {
    "explanation": "<explanation>",
    "overall": <1-5>
    }
    '''

    examples = '''
    Here are some examples:
    Example 1:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: BSTs let you perform searches in O(log n) time.
    Output: {
        "explanation": "No example, no steps, no analogy.",
        "overall": 1
        }
    Example 2:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: BSTs let you search, e.g., finding a student ID quickly.
    Output: {
        "explanation": "One brief example only.",
        "overall": 2
        }
    Example 3:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: Step 1: Start at root. Step 2: Compare key. Step 3: Move left or right until found.
    Output: {
        "explanation": "Full step list but no example/analogy.",
        "overall": 3
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: Imagine a phone book sorted alphabetically (analogy). Step 1… Step 3… Finally, for example, you can locate roll-number 73 in ~7 comparisons.
    Output: {
        "explanation": "Two devices (analogy + steps) with good clarity.",
        "overall": 4
        }
    Example 4:
    Input: [Q]: Explain what a Binary Search Tree (BST) is and why it is useful? Answer: “Think of a BST like a decision fork in a 20‑questions game (analogy). Example: searching 42 follows arrows in the diagram below. Steps: 1 Start at 50, 2 go left… Visual mention: 'See diagram'.”
    Output: {
        "explanation": "Analogy + example + step list (+ visual cue) ⇒ three devices, all complete.",
        "overall": 5
        }
        
    '''

    QUERY_PROMPT = "Given the following question and answers, assign the appropriate score and give the explanation as shown in the examples. Output the dictionary format as described above. Do not include any other text. "
    final_prompt = SYSTEM_PROMPT + TASK_INSTRUCTION + OUTPUT_FORMAT + examples + QUERY_PROMPT + "[Q]: " + question + "Answer: " + generated + " Output: " #change
    response = gpt4_response(final_prompt) #can change to gemini_response if needed
    #print(response)
    # Extract the overall score from the response
    extracted_json = extract_json(response)
    score = json.loads(extracted_json)["overall"]
    return score

def get_all_scores(question, answer, generated):
    if config["metric choices"]["entailment score"]:
        entailment_score = entailment(question, answer, generated)
    else:
        entailment_score = None
    #entailment_score = entailment(question, answer, generated)
    if config["metric choices"]["factqa score"]:
        factqa_p, factqa_r = factqa(question, answer, generated)
    else:
        factqa_p, factqa_r = None, None
    #factqa_p, factqa_r = factqa(question, answer, generated)
    if config["metric choices"]["clarity"]:
        clarity_score = clarity(question, answer, generated)
    else:
        clarity_score = None
    #clarity_score = clarity(question, answer, generated)
    if config["metric choices"]["encouraging critical thinking"]:
        critical_score = critical(question, answer, generated)
    else:
        critical_score = None
    #critical_score = critical(question, answer, generated)
    if config["metric choices"]["using pedagogical techniques"]:
        pedagogical_score = pedagogical(question, answer, generated)
    else:
        pedagogical_score = None
    #pedagogical_score = pedagogical(question, answer, generated)
    '''
    print(f"Entailment Score: {entailment_score}")
    print(f"FactQA Precision: {factqa_p}, Recall: {factqa_r}")
    print(f"Clarity Score: {clarity_score}")
    print(f"Encouraging Critical Thinking Score: {critical_score}")
    print(f"Using Pedagogical Techniques Score: {pedagogical_score}")
    '''
    return entailment_score, factqa_p, factqa_r, clarity_score, critical_score, pedagogical_score