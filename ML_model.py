import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class HangmanDataset(Dataset):
    def __init__(self, dictionary_file, num_samples=100000, max_word_length=20):
        self.max_word_length = max_word_length
        with open(dictionary_file, 'r') as f:
            self.words = [word.strip().lower() for word in f if 3 <= len(word.strip()) <= max_word_length]
        
        self.samples = []
        self.generate_training_samples(num_samples)
        
    def generate_training_samples(self, num_samples):
        for _ in range(num_samples):
            word = random.choice(self.words)
            revealed = ['_'] * len(word)
            guessed_letters = set()
            
            num_revealed = random.randint(0, len(word)-1)
            positions = random.sample(range(len(word)), num_revealed)
            
            for pos in positions:
                revealed[pos] = word[pos]
                guessed_letters.add(word[pos])
            
            remaining_letters = set(word) - guessed_letters
            if remaining_letters:
                target_letter = random.choice(list(remaining_letters))
                current_state = ''.join(revealed).ljust(self.max_word_length, '_')[:self.max_word_length]
                self.samples.append((current_state, list(guessed_letters), target_letter))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        current_state, guessed_letters, target_letter = self.samples[idx]
        
        # Convert to features with fixed size
        state_features = torch.zeros(self.max_word_length, 26)
        guessed_features = torch.zeros(26)
        
        # Encode current state
        for i, char in enumerate(current_state):
            if char != '_':
                state_features[i][ord(char) - ord('a')] = 1
                
        # Encode guessed letters
        for letter in guessed_letters:
            guessed_features[ord(letter) - ord('a')] = 1
            
        # Encode target
        target = torch.zeros(26)
        target[ord(target_letter) - ord('a')] = 1
        
        return {
            'state': state_features,
            'guessed': guessed_features,
            'target': target
        }

def collate_fn(batch):
    max_len = max(item['state'].shape[0] for item in batch)
    
    padded_batch = {
        'state': torch.stack([
            torch.cat([item['state'], 
                      torch.zeros(max_len - item['state'].shape[0], 26)], dim=0)
            for item in batch
        ]),
        'guessed': torch.stack([item['guessed'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }
    return padded_batch

class HangmanModel(nn.Module):
    def __init__(self, max_word_length=20):
        super(HangmanModel, self).__init__()
        self.max_word_length = max_word_length
        
        self.state_encoder = nn.Sequential(
            nn.Linear(max_word_length * 26, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.guessed_encoder = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.combined = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 26),
            nn.Sigmoid()
        )
        
    def forward(self, state, guessed):
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        state_features = self.state_encoder(state)
        guessed_features = self.guessed_encoder(guessed)
        combined = torch.cat([state_features, guessed_features], dim=1)
        return self.combined(combined)

def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            guessed = batch['guessed'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(state, guessed)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                guessed = batch['guessed'].to(device)
                target = batch['target'].to(device)
                
                output = model(state, guessed)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                target_idx = target.argmax(dim=1)
                correct += (pred == target_idx).sum().item()
                total += target.size(0)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {100*correct/total:.2f}%')
        print('--------------------')

def predict_letter(model, current_state, guessed_letters, device, max_word_length=20):
    model.eval()
    
    # Pad input to max_word_length
    padded_state = current_state.ljust(max_word_length, '_')[:max_word_length]
    
    state_features = torch.zeros(1, max_word_length, 26)
    guessed_features = torch.zeros(1, 26)
    
    for i, char in enumerate(padded_state):
        if char != '_':
            state_features[0, i, ord(char) - ord('a')] = 1
            
    for letter in guessed_letters:
        guessed_features[0, ord(letter) - ord('a')] = 1
    
    state_features = state_features.to(device)
    guessed_features = guessed_features.to(device)
    
    with torch.no_grad():
        output = model(state_features, guessed_features)
        
    for letter in guessed_letters:
        output[0, ord(letter) - ord('a')] = -float('inf')
    
    letter_idx = output.argmax(dim=1)
    return chr(letter_idx.item() + ord('a'))

def main():
    # Configuration
    max_word_length = 20
    num_epochs = 128
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = HangmanDataset(
        "words_250000_train.txt", 
        num_samples=1000000,
        max_word_length=max_word_length
    )
    val_dataset = HangmanDataset(
        "words_250000_train.txt",
        num_samples=200000,
        max_word_length=max_word_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = HangmanModel(max_word_length=max_word_length).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Save model
    torch.save(model.state_dict(), 'hangman_model.pth')
    
    # Example prediction
    example_state = "a__l_"
    guessed_letters = {'a', 'l'}
    prediction = predict_letter(
        model=model,
        current_state=example_state,
        guessed_letters=guessed_letters,
        device=device,
        max_word_length=max_word_length
    )
    print(f"Current state: {example_state}")
    print(f"Guessed letters: {guessed_letters}")
    print(f"Model prediction: {prediction}")

if __name__ == "__main__":
    main()