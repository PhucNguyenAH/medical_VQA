from abc import abstractmethod

from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        pass

def sample_qa(conversation):
    # Ensure the conversation length is even
    if len(conversation) % 2 != 0:
        raise ValueError("Conversation length must be even.")

    # Randomly select an even index (question index)
    max_index = len(conversation) - 2  # Last valid index for a question
    question_index = random.choice(
        range(0, max_index + 1, 2)
    )  # Select from even indices

    # Get the corresponding answer (next index)
    answer_index = question_index + 1

    # Extract the question and answer
    question = conversation[question_index]["value"]
    answer = conversation[answer_index]["value"]

    return question, answer


class LLaVAInstruct150Dataset(VQADataset):
    def __init__(
        self,
        data_dir="./llava_instruct_data/llava_v1_5_mix665k_cleaned.json",
        image_dir="llava_image_data",
    ):
        """
        Args:
            data (list of dict): The dataset, where each entry is a dictionary containing 'conversations', 'id', and 'image'.
            processor (callable): The processor function or object to process the image and text data.
            image_dir (str): The directory where the images are stored.
        #"""
        self.image_dir = image_dir
        self.images = set()
        self.prompt = "<VQA>"
        with open(data_dir) as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Sample a QA pair from the conversation
        question, answer = sample_qa(row["conversations"])

        # Prepend the prompt to the question
        question = self.prompt + question

        # Load and convert the image
        image = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")

        return question, answer, image

    def __len__(self):
        return len(self.data)