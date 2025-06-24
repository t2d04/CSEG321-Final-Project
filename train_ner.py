import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from seqeval.metrics import f1_score, classification_report

from models.gpt2 import GPT2Model
from optimizer import AdamW
from torch import nn

import warnings
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
# Set up key parameters and file paths
# Make sure your data files are in the same directory as this script
TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"
MODEL_NAME = 'gpt2'
NUM_EPOCHS = 20 # You can reduce this to 1 or 2 if you are short on time
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 5e-5

# --- 2. Create Label to ID Mappings ---
# Create a mapping from your string tags to integer IDs and back
# The order matters! 'O' should be 0.
unique_tags = sorted(list(set([
    'O', 'B-MAT', 'I-MAT', 'B-QUANT', 'I-QUANT', 'B-PURE', 'I-PURE', 
    'B-ACT', 'I-ACT', 'B-OBJ', 'I-OBJ', 'B-APP', 'I-APP', 
    'B-OTHC', 'I-OTHC', 'B-TIME', 'I-TIME'
])))

label_to_id = {tag: id for id, tag in enumerate(unique_tags)}
id_to_label = {id: tag for id, tag in enumerate(unique_tags)}
NUM_LABELS = len(unique_tags)


# --- 3. Custom Dataset for NER ---
# This class reads our .txt file and prepares it for the model
class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_id, max_length):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.sentences, self.labels = self._read_data(file_path)

    def _read_data(self, file_path):
        sentences, labels = [], []
        current_sentence, current_labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence, current_labels = [], []
                else:
                    try:
                        word, tag = line.split()
                        current_sentence.append(word)
                        current_labels.append(tag)
                    except ValueError:
                        print(f"Skipping malformed line: {line}")
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, labels = self.sentences[idx], self.labels[idx]
        
        # Tokenize words and align labels
        tokenized_inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Ignore special tokens [CLS], [SEP], [PAD]
            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[labels[word_idx]]) # Label for the first token of a word
            else:
                label_ids.append(-100) # Ignore subsequent tokens of the same word
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = torch.tensor(label_ids, dtype=torch.long)
        tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.long)
        
        return tokenized_inputs

# --- 4. Custom Model for Token Classification ---
# This class puts a classification head on top of our GPT-2 model
class GPT2ForTokenClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        # Load our custom GPT-2 implementation
        self.gpt2 = GPT2Model.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(self.gpt2.config.hidden_dropout_prob)
        # The final layer that maps hidden states to our NER tags
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        sequence_output = outputs['last_hidden_state']
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # CrossEntropyLoss automatically ignores indices where label is -100
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits

# --- 5. Evaluation Function ---
def evaluate(model, dataloader, device, id_to_label):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            preds = torch.argmax(logits, dim=2)

            for i in range(labels.shape[0]):
                true_seq = []
                pred_seq = []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        true_seq.append(id_to_label[labels[i, j].item()])
                        pred_seq.append(id_to_label[preds[i, j].item()])
                all_labels.append(true_seq)
                all_preds.append(pred_seq)
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds)
    
    return f1, report


# --- 6. Main Training and Evaluation Script ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
    # GPT-2 tokenizer doesn't have a default pad token, set it to the end-of-sequence token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets and dataloaders
    train_dataset = NERDataset(TRAIN_FILE, tokenizer, label_to_id, MAX_LENGTH)
    dev_dataset = NERDataset(DEV_FILE, tokenizer, label_to_id, MAX_LENGTH)
    test_dataset = NERDataset(TEST_FILE, tokenizer, label_to_id, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model, optimizer
    model = GPT2ForTokenClassification(num_labels=NUM_LABELS).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Evaluate on dev set after each epoch
        dev_f1, dev_report = evaluate(model, dev_loader, device, id_to_label)
        print(f"\n--- Epoch {epoch+1} Dev Set Evaluation ---")
        print(f"Macro F1 Score: {dev_f1:.4f}")
        print(dev_report)
        print("-------------------------------------\n")

    print("--- Training Finished ---")
    
    # Final evaluation on the test set
    test_f1, test_report = evaluate(model, test_loader, device, id_to_label)
    print("\n\n--- Final Test Set Evaluation ---")
    print(f"Macro F1 Score: {test_f1:.4f}")
    print("This is the final score for your presentation!")
    print(test_report)


if __name__ == "__main__":
    main()