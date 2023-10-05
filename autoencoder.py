import torch
import torch.nn as nn
import torch.optim as optim

# Convert your data to PyTorch tensors


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, code_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Use sigmoid for binary cross-entropy loss
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(X_train):
    
    X_train_tensor = torch.FloatTensor(X_train.toarray())
    input_size = X_train.shape[1]
    hidden_size = 128
    code_size = 64
    learning_rate = 0.001
    batch_size = 256
    epochs = 60

    # Initialize the model, loss, and optimizer
    model = Autoencoder(input_size, hidden_size, code_size, batch_size)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training complete!")
    return model

def reconstruction_error(model, X_test):
    

    # Convert test data to tensor
    X_test_tensor = torch.FloatTensor(X_test.toarray())

    # Compute reconstruction error on test data
    model.eval()
    with torch.no_grad():
        reconstructed_test = model(X_test_tensor)
        test_error = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
    return test_error