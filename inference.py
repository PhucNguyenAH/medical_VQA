from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
import torch
import numpy as np

WEIGHT = "SonicNLP/Florence-2-FT-MedVQA"
DATASET = "SonicNLP/medvqa_dataset"
data = load_dataset(DATASET)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

code_rev = "refs/pr/6"
base_cfg = AutoConfig.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision=code_rev,      
)

# model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6').to(device)
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')
model = AutoModelForCausalLM.from_pretrained(WEIGHT, trust_remote_code=True, revision="main", code_revision=code_rev, config=base_cfg).to(device)
processor = AutoProcessor.from_pretrained(WEIGHT, trust_remote_code=True, revision="main", code_revision="refs/pr/6")

# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input
    print("Prompt:", prompt)

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

for idx in range(3):
    print(run_example("<VQA>", data['test'][idx]['question'], data['test'][idx]['image']))
    print("Ground Truth Answers:", data['test'][idx]['answers'])
    # display(data['train'][idx]['image'].resize([350, 350]))

# print("Keys: ", data['train'][0].keys())
# # print("questionId:", data['train'][0]['questionId'])
# print("question:", data['train'][0]['question'])
# # print("question_types:", data['train'][0]['question_types'])
# image = data['train'][0]['image']
# if image.mode != "RGB":
#     image = image.convert("RGB")
# img = np.array(image)
# print("image:",type(image), img.shape)
# # print("ucsf_document_id:", data['train'][0]['ucsf_document_id'])
# # print("ucsf_document_page_no:", data['train'][0]['ucsf_document_page_no'])
# print("answers:", data['train'][0]['answers'])