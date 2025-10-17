import os
import json
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Step 1: Build the teacher model
teacher_model = LLM(
    model="./models/Phi-3-vision-128k-instruct",
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_num_seqs=4,
)

# Define general configurations
sampling_params = SamplingParams(temperature=0, max_tokens=1024)
data_path = "datasets/llava_instruct_data/llava_v1_5_mix665k.json"
image_dir = "datasets/llava_image_data/"
output_dir = "datasets/llava_instruct_data/"
output_filename= "llava_v1_5_cleaned_refine.json"

def process_image(image_path):
    """Load and preprocess an image."""
    return Image.open(image_path).convert("RGB")

def generate_question(teacher_model, image, attempt):
    """Generate a question for the given image using the teacher model."""
    prompt = f"<|user|>\n<|image_1|>\nGenerate a question from this image. You attempt {attempt} times.<|end|>\n<|assistant|>\n"
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }
    outputs = teacher_model.generate([inputs], sampling_params=sampling_params)
    return outputs[0].outputs[0].text

def generate_answer(teacher_model, image, question):
    """Generate an answer for the given question and image using the teacher model."""
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }
    outputs = teacher_model.generate([inputs], sampling_params=sampling_params)
    return outputs[0].outputs[0].text

# Load dataset
with open(data_path) as f:
    dataset = json.load(f)


# Process dataset
for i, row in tqdm(enumerate(dataset), desc="Processing data"):
    image_path = os.path.join(image_dir, row['image'])
    image = process_image(image_path)

    # Step 2: Generate a question for the image
    question = generate_question(teacher_model, image, attempt=i + 1)

    # Step 3: Generate an answer based on the question and image
    answer = generate_answer(teacher_model, image, question)

    # Append the generated question and answer to the dataset
    row['conversations'].append({"from": "teacher_model_question", "value": question})
    row['conversations'].append({"from": "teacher_model_answer", "value": answer})

output_file = os.path.join(output_dir, output_filename)
with open(output_file, "w") as fp:
    json.dump(dataset, fp)

print("Distillation process complete.")