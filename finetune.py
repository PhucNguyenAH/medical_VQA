import os
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoProcessor, get_scheduler, get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(gradient_as_bucket_view=False)
LOGDIR = os.path.abspath("./runs")
os.makedirs(LOGDIR, exist_ok=True)
accelerator = Accelerator(
    log_with="tensorboard",
    project_dir=LOGDIR,
    kwargs_handlers=[ddp_kwargs],
)


class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<VQA>" + example['question']
        first_answer = example['answers'][0]
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    return inputs, answers

def train_model(train_loader, val_loader, model, processor, epochs=1, lr=1e-6, log_every=10):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    # lr_scheduler = get_scheduler("linear",
    #                              optimizer=optimizer,
    #                              num_warmup_steps=0,
    #                              num_training_steps=num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                 num_warmup_steps=len(train_loader),
                                 num_training_steps=num_training_steps)

    # Prepare everything for distributed / mixed precision
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Init tracker (creates ./runs/florence2-medvqa-… for TensorBoard)
    accelerator.init_trackers(
        project_name="florence2-medvqa",
        config={
            "epochs": epochs,
            "lr": lr,
            "train_batch_size": train_loader.batch_size,
            "val_batch_size": val_loader.batch_size,
            "model": "microsoft/Florence-2-base-ft",
        },
    )

    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        # ------------------ TRAIN ------------------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process,
                    desc=f"Train {epoch+1}/{epochs}")
        for step, (inputs, answers) in enumerate(pbar):
            input_ids    = inputs["input_ids"].to(device, non_blocking=True)
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
            labels = processor.tokenizer(text=answers,
                                         return_tensors="pt",
                                         padding=True,
                                         return_token_type_ids=False).input_ids.to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            running += loss.detach().float().item()
            global_step += 1

            # ---- logging
            if (step + 1) % log_every == 0:
                accelerator.log(
                    {"train/loss": loss.detach().float().item(),
                     "train/lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step
                )
                if accelerator.is_local_main_process:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        # average train loss across processes
        running = torch.tensor(running, device=device)
        count   = torch.tensor(len(train_loader), device=device, dtype=torch.float32)
        running = accelerator.reduce(running, reduction="mean")
        count   = accelerator.reduce(count, reduction="mean")
        avg_train = (running / count).item()
        accelerator.print(f"Average Training Loss: {avg_train:.6f}")
        accelerator.log({"train/epoch_loss": avg_train, "epoch": epoch+1}, step=global_step)

        # ------------------ VAL ------------------
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, disable=not accelerator.is_local_main_process,
                        desc=f"Val {epoch+1}/{epochs}")
            for inputs, answers in pbar:
                input_ids    = inputs["input_ids"].to(device, non_blocking=True)
                pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
                labels = processor.tokenizer(text=answers,
                                             return_tensors="pt",
                                             padding=True,
                                             return_token_type_ids=False).input_ids.to(device, non_blocking=True)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                running_val += outputs.loss.detach().float().item()

        running_val = torch.tensor(running_val, device=device)
        count_val   = torch.tensor(len(val_loader), device=device, dtype=torch.float32)
        running_val = accelerator.reduce(running_val, reduction="mean")
        count_val   = accelerator.reduce(count_val, reduction="mean")
        avg_val = (running_val / count_val).item()
        accelerator.print(f"Average Validation Loss: {avg_val:.6f}")
        accelerator.log({"val/epoch_loss": avg_val, "epoch": epoch+1}, step=global_step)

        # ------------------ SAVE ------------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            if accelerator.is_main_process:
                accelerator.print(f"New best checkpoint at epoch {epoch+1} (val_loss={avg_val:.6f})")

                best_dir = f"./model_checkpoints/best"
                os.makedirs(best_dir, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                unwrapped.config = base_cfg   # ensure vision_config & auto_map are saved
                unwrapped.save_pretrained(best_dir, safe_serialization=True)
                if hasattr(unwrapped, "generation_config") and unwrapped.generation_config is not None:
                    unwrapped.generation_config.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
        
        if epoch % 20 == 0 and epoch > 0:
            if accelerator.is_main_process:
                checkpoint_dir = f"./model_checkpoints/epoch_{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                unwrapped.config = base_cfg   # ensure vision_config & auto_map are saved
                unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True)
                if hasattr(unwrapped, "generation_config") and unwrapped.generation_config is not None:
                    unwrapped.generation_config.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)

    accelerator.end_training()

HF_CKP = "SonicNLP/Florence-2-FT-MedVQA"
CODE_REV = "refs/pr/6"
EPOCHS = 100
batch_size = 5
num_workers = 0
learning_rate = 1e-6

# data = load_dataset("HuggingFaceM4/DocumentVQA")
data = load_dataset("SonicNLP/medvqa_dataset")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = accelerator.device

base_cfg = AutoConfig.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision=CODE_REV,
)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision=CODE_REV, config=base_cfg).to(device)
# model = torch.nn.DataParallel(model) 
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision=CODE_REV)

# Create datasets
train_dataset = DocVQADataset(data['train'])
val_dataset = DocVQADataset(data['validation'])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=learning_rate)
model.push_to_hub(HF_CKP, private=True)
processor.push_to_hub(HF_CKP, private=True)