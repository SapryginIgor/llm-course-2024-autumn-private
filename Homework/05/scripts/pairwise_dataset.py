import torch
from torch.utils.data import Dataset
import torch
import transformers
import datasets
import trl


class IMDBPairwiseDataset(Dataset):
    """ 
    A dataset of all possible pairs of chosen and rejected texts for TRL reward training format.

    This dataset is designed to facilitate the training of a reward model by providing pairs of
    texts where one is preferred (chosen) and the other is not (rejected). Each sample in the dataset
    is a dictionary containing tokenized input IDs and attention masks for both the chosen and rejected
    texts.

    Parameters:
    imdb: dataset to pairwise
    tokenizer: The tokenizer used to preprocess the texts
    accepted_label (int): The label that indicates a chosen text. Texts with this label are considered
                          preferred, while others are considered rejected.

    Methods:
    __len__(): Returns the total number of possible pairs of chosen and rejected texts.
    __getitem__(index): Returns a dictionary containing tokenized inputs for a specific pair of chosen
                        and rejected texts.
    """
    
    def __init__(self, imdb, tokenizer, accepted_label):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = [x['text'] for x in imdb if x['label'] == accepted_label]
        self.rejected_texts = [x['text'] for x in imdb if x['label'] != accepted_label]

        assert self.chosen_texts, f"no texts with label {accepted_label}"
        # print(f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs")

        self.column_names = [
            'input_ids_chosen', 'attention_mask_chosen',
            'input_ids_rejected', 'attention_mask_rejected'
        ]

    def __len__(self):
        return len(self.chosen_texts)*len(self.rejected_texts)

    def __getitem__(self, index: int):
        batch_chosen= self.tokenizer(self.chosen_texts[index//len(self.rejected_texts)], return_attention_mask=True)
        chosen, chosen_attention = batch_chosen['input_ids'], batch_chosen['attention_mask']
        batch_rejected = self.tokenizer(self.rejected_texts[index%len(self.rejected_texts)], return_attention_mask=True)
        rejected, rejected_attention = batch_rejected['input_ids'], batch_rejected['attention_mask']
        return dict(
            input_ids_chosen=chosen,
            attention_mask_chosen=chosen_attention,
            input_ids_rejected=rejected,
            attention_mask_rejected=rejected_attention
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_tokenizer = transformers.AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    main_model = transformers.AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb", device_map=device)
    inputs = main_tokenizer("The movie", return_tensors='pt').to(device)
    generated_ids = main_model.generate(**inputs, max_new_tokens=50, do_sample=True)
    reward_model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased",
                                                                                   device_map=device)
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-cased")
    TARGET_LABEL = 0  # negative reviews
    imdb = datasets.load_dataset("imdb", split='train')
    reward_data = IMDBPairwiseDataset(imdb, reward_tokenizer, accepted_label=TARGET_LABEL)
    training_args = trl.RewardConfig(  # like transformers.TrainingArguments
        output_dir="reward_model",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1.41e-5,
        max_steps=1_000,  # note: training may need more than 1k steps
        logging_steps=50,
        gradient_checkpointing=True,  # reduce memory usage but train ~30% slower
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,  # disable this on CPU or on very old GPUs
        report_to='none',
        # you may add any other hyperparameters that you found useful
    )

    trainer = trl.RewardTrainer(
        model=reward_model,
        args=training_args,
        tokenizer=reward_tokenizer,
        train_dataset=reward_data,
        peft_config=None,  # optionally, you may tune with LoRA, prompt-tuning, etc
    )

    trainer.train()