# Step 1: Enhance Dataset Class for Synthetic Data Generation


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from typing import List, Dict
from torch.utils.data import Dataset

class SyntheticDataGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    def generate(self, prompt: str, max_length: int = 200, num_samples: int = 5) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_samples,
            temperature=0.7
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


class EnhancedSFTDataset(Dataset):
    def __init__(self, file: str, tokenizer, max_seq_length: int, template: Dict, synthetic_generator: SyntheticDataGenerator = None):
        self.tokenizer = tokenizer
        self.template = template
        self.max_seq_length = max_seq_length
        self.synthetic_generator = synthetic_generator
        
        with open(file, "r", encoding="utf8") as f:
            data_list = f.readlines()
        
        if synthetic_generator:
            for data in data_list:
                json_data = json.loads(data)
                if "conversations" in json_data:
                    last_conversation = json_data["conversations"][-1]["content"]
                    synthetic_responses = synthetic_generator.generate(last_conversation)
                    for response in synthetic_responses:
                        synthetic_data = json_data.copy()
                        synthetic_data["conversations"].append({"role": "assistant", "content": response})
                        data_list.append(json.dumps(synthetic_data))
        
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = json.loads(self.data_list[index])
        input_ids, target_mask = [], []

        if "system" in data:
            system_text = self.template["system_format"].format(content=data["system"].strip())
            input_ids.extend(self.tokenizer.encode(system_text, add_special_tokens=False))
            target_mask.extend([0] * len(input_ids))

        for i in range(0, len(data["conversations"]) - 1, 2):
            if data["conversations"][i]["role"] == "user" and data["conversations"][i + 1]["role"] == "assistant":
                user_text = self.template["user_format"].format(
                    content=data["conversations"][i]["content"].strip()
                )
                assistant_text = self.template["assistant_format"].format(
                    content=data["conversations"][i + 1]["content"].strip()
                )

                user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
                assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)

                input_ids.extend(user_ids + assistant_ids)
                target_mask.extend([0] * len(user_ids) + [1] * len(assistant_ids))

        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        return inputs
