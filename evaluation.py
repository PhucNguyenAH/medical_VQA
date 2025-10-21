import evaluate
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
import torch
from tqdm import tqdm

def compute_metrics(predictions, references, use_bertscore=False):
    # Load evaluation metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore") if use_bertscore else None

    # --- ROUGE ---
    rouge_result = rouge.compute(predictions=predictions, references=references)
    print("\n🟥 ROUGE:")
    for k, v in rouge_result.items():
        print(f"{k}: {v:.4f}")

    # --- BLEU ---
    bleu_result = bleu.compute(predictions=predictions, references=references)
    print("\n🔷 BLEU:")
    print(f"BLEU score: {bleu_result['bleu']:.4f}")

    # --- METEOR ---
    meteor_result = meteor.compute(predictions=predictions, references=references)
    print("\n🟡 METEOR:")
    print(f"METEOR score: {meteor_result['meteor']:.4f}")

    # --- BERTScore (optional) ---
    avg_f1 = None
    if use_bertscore:
        bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
        avg_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
        print("\n🟢 BERTScore:")
        print(f"F1: {avg_f1:.4f}")

    return {
        "rouge": rouge_result,
        "bleu": bleu_result,
        "meteor": meteor_result,
        "bertscore": bert_result,
    }
PRETRAINED = "microsoft/Florence-2-base-ft"
WEIGHT = "SonicNLP/Florence-2-FT-MedVQA"

DATASET = "SonicNLP/medvqa_dataset"
data = load_dataset(DATASET)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_rev = "refs/pr/6"
base_cfg = AutoConfig.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision=code_rev,      
)
pretrained_model = AutoModelForCausalLM.from_pretrained(PRETRAINED, trust_remote_code=True, revision=code_rev).to(device)
pretrained_processor = AutoProcessor.from_pretrained(PRETRAINED, trust_remote_code=True, revision=code_rev)
model = AutoModelForCausalLM.from_pretrained(WEIGHT, trust_remote_code=True, config=base_cfg).to(device)
processor = AutoProcessor.from_pretrained(WEIGHT, trust_remote_code=True)

# Generate predictions for the test set
pretrained_predictions = []
predictions = []
references = []
for idx in tqdm(range(len(data['test']))):
    question = data['test'][idx]['question']
    image = data['test'][idx]['image']
    ground_truth_answers = data['test'][idx]['answers']

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text="<VQA>" + question, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<VQA>", image_size=(image.width, image.height))
    predictions.append(parsed_answer['<VQA>'])

    generated_ids = pretrained_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = pretrained_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = pretrained_processor.post_process_generation(generated_text, task="<VQA>", image_size=(image.width, image.height))

    pretrained_predictions.append(parsed_answer['<VQA>'])
    references.append(ground_truth_answers[0])  # Assuming the first answer is the reference

# Compute and print evaluation metrics
compute_metrics(pretrained_predictions, references, use_bertscore=True)
compute_metrics(predictions, references, use_bertscore=True)