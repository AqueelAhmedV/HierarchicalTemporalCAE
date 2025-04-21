import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Initialize weights for the forget gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        # Initialize weights for the input gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        # Initialize weights for the cell gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        
        # Initialize weights for the output gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        
        # Initialize weights for the final output
        self.Wy = np.random.randn(input_size, hidden_size) * 0.01
        self.by = np.zeros((input_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.vstack((x, h_prev))
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        
        # Cell gate
        c_hat = self.tanh(np.dot(self.Wc, combined) + self.bc)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # New cell state
        c_next = f * c_prev + i * c_hat
        
        # New hidden state
        h_next = o * self.tanh(c_next)
        
        # Output
        y = np.dot(self.Wy, h_next) + self.by
        
        return y, h_next, c_next

# Example usage
def main():
    # Create a simple sequence prediction problem
    # We'll try to predict the next number in a simple pattern: [0, 1, 2, 3, 4, ...]
    
    # Initialize LSTM
    input_size = 1
    hidden_size = 4
    lstm = LSTM(input_size, hidden_size)
    
    # Create training data
    sequence = np.array([i for i in range(10)]).reshape(-1, 1)
    
    # Training parameters
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    print("Sequence prediction example:")
    print("Input sequence:", sequence.flatten())
    
    # Forward pass through the sequence
    predictions = []
    for i in range(len(sequence) - 1):
        x = sequence[i].reshape(-1, 1)
        y, h, c = lstm.forward(x, h, c)
        predictions.append(y[0, 0])
    
    print("Predicted next values:", np.array(predictions))
    print("Actual next values:", sequence[1:].flatten())

if __name__ == "__main__":
    main()
