# --- TROUBLESHOOTING: "Run Python" not working in Visual Studio Code ---
# 1. Make sure you have the Python extension installed in VS Code.
# 2. Check that Python is installed and available in your system PATH.
# 3. Select the correct Python interpreter (Ctrl+Shift+P > "Python: Select Interpreter").
# 4. If using a virtual environment, activate it in the terminal or select it in VS Code.
# 5. Try running your script from the integrated terminal: `python /home/sean/pybit/pybit.py`
# 6. Restart VS Code if changes don't take effect.
# 7. Check for errors in the "PROBLEMS" or "OUTPUT" panel.
# 8. If issues persist, reinstall the Python extension or VS Code.

import numpy as np
import string

# A simple, custom-built neural network for character prediction.
class SimpleNeuralNet:
    def save_weights(self, filename):
        """Save weights and biases to a file."""
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def load_weights(self, filename):
        """Load weights and biases from a file if it exists."""
        try:
            data = np.load(filename)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.W4 = data['W4']
            self.b4 = data['b4']
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}, starting fresh.")
    """
    A simple feed-forward neural network for sequence prediction with 3 hidden layers.
    It learns patterns from a training dataset using backpropagation.
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, dropout_rate=0.2):
        # Initialize weights and biases for three hidden layers.
        self.dropout_rate = dropout_rate
        self.W1 = np.random.randn(input_size, hidden_size_1) * 0.01
        self.b1 = np.zeros((1, hidden_size_1))
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.01
        self.b2 = np.zeros((1, hidden_size_2))
        self.W3 = np.random.randn(hidden_size_2, hidden_size_3) * 0.01
        self.b3 = np.zeros((1, hidden_size_3))
        self.W4 = np.random.randn(hidden_size_3, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

    def _sigmoid(self, x):
        # The sigmoid activation function. It squashes values between 0 and 1.
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # The derivative of the sigmoid function, used for backpropagation.
        return x * (1 - x)

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Trains the neural network using gradient descent.
        """
        print("Starting training...")
        for epoch in range(epochs):
            # Forward propagation through three hidden layers
            z1 = np.dot(X_train, self.W1) + self.b1

            a1 = self._sigmoid(z1)
            # Dropout for first hidden layer
            dropout_mask1 = (np.random.rand(*a1.shape) > self.dropout_rate).astype(float)
            a1 *= dropout_mask1

            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)
            # Dropout for second hidden layer
            dropout_mask2 = (np.random.rand(*a2.shape) > self.dropout_rate).astype(float)
            a2 *= dropout_mask2

            z3 = np.dot(a2, self.W3) + self.b3
            a3 = self._sigmoid(z3)
            # Dropout for third hidden layer
            dropout_mask3 = (np.random.rand(*a3.shape) > self.dropout_rate).astype(float)
            a3 *= dropout_mask3

            z4 = np.dot(a3, self.W4) + self.b4
            a4 = self._sigmoid(z4)

            # Backpropagation (error calculation)
            error = y_train - a4

            # Use the error to find the gradients and update weights
            d_a4 = error * self._sigmoid_derivative(a4)
            d_W4 = np.dot(a3.T, d_a4)
            d_b4 = np.sum(d_a4, axis=0, keepdims=True)

            d_a3 = np.dot(d_a4, self.W4.T) * self._sigmoid_derivative(a3)
            d_W3 = np.dot(a2.T, d_a3)
            d_b3 = np.sum(d_a3, axis=0)

            d_a2 = np.dot(d_a3, self.W3.T) * self._sigmoid_derivative(a2)
            d_W2 = np.dot(a1.T, d_a2)
            d_b2 = np.sum(d_a2, axis=0)

            d_a1 = np.dot(d_a2, self.W2.T) * self._sigmoid_derivative(a1)
            d_W1 = np.dot(X_train.T, d_a1)
            d_b1 = np.sum(d_a1, axis=0)

            # Update weights and biases
            self.W1 += d_W1 * learning_rate
            self.b1 += d_b1 * learning_rate
            self.W2 += d_W2 * learning_rate
            self.b2 += d_b2 * learning_rate
            self.W3 += d_W3 * learning_rate
            self.b3 += d_b3 * learning_rate
            self.W4 += d_W4 * learning_rate
            self.b4 += d_b4 * learning_rate

            if (epoch + 1) % 5000 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
                self.save_weights("model_weights.npz")

        print("Training complete!")

    def predict(self, X_test):
        """
        Predicts the output for new input data.
        Returns output activations as probabilities for each character.
        """
        z1 = np.dot(X_test, self.W1) + self.b1
        a1 = self._sigmoid(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._sigmoid(z3)

        z4 = np.dot(a3, self.W4) + self.b4
        a4 = self._sigmoid(z4)

        # Normalize output activations to sum to 1 (softmax-like sampling)
        probs = a4[0] / np.sum(a4[0])
        # Sample a character index based on probabilities
        sampled_index = np.random.choice(len(probs), p=probs)
        sampled_vector = np.zeros_like(probs)
        sampled_vector[sampled_index] = 1
        return np.array([sampled_vector])

# --- One-hot encoding and decoding functions ---
def char_to_one_hot(char, alphabet):
    """Converts a character to a one-hot encoded vector."""
    vector = np.zeros(len(alphabet))
    vector[alphabet.find(char)] = 1
    return vector

def one_hot_to_char(vector, alphabet):
    """Converts a one-hot encoded vector back to a character."""
    index = np.argmax(vector)
    return alphabet[index]

def pad_sequence(sequence, max_len, alphabet):
    """Pads a sequence with spaces to a fixed length."""
    if len(sequence) > max_len:
        sequence = sequence[-max_len:]
    padded_sequence = ' ' * (max_len - len(sequence)) + sequence
    
    # One-hot encode each character and concatenate
    one_hot_input = np.concatenate([char_to_one_hot(c, alphabet) for c in padded_sequence])
    return one_hot_input

def generate_sequence(model, start_sequence, length, max_len, alphabet):
    """
    Generates a new sequence of a given length using the trained model.
    The model is used autoregressively.
    
    Args:
        model (SimpleNeuralNet): The trained neural network model.
        start_sequence (str): The initial string to start the generation.
        length (int): The number of characters to generate.
        max_len (int): The maximum length of the input sequence for the model.
        alphabet (str): The alphabet used for encoding/decoding.
        
    Returns:
        str: The generated sequence.
    """
    generated_sequence = start_sequence
    for _ in range(length):
        # Get the latest part of the sequence to feed to the model
        input_for_prediction = generated_sequence[-max_len:]
        
        # Pad and one-hot encode the input
        one_hot_input = pad_sequence(input_for_prediction, max_len, alphabet)
        input_for_model = np.array([one_hot_input])
        
        # Predict the next character
        prediction_vector = model.predict(input_for_model)
        predicted_char = one_hot_to_char(prediction_vector[0], alphabet)
        
        # Append the new character to the generated sequence
        generated_sequence += predicted_char
        
    return generated_sequence[len(start_sequence):]


# --- Main Program Execution ---
if __name__ == "__main__":
    
    # 1. Prepare the alphabet and training data
    # Load training data from a file
    with open("training_data.txt", "r", encoding="utf-8") as f:
        training_sequence_text = f.read()
    # Replace carriage returns and line feeds with a space
    training_sequence_text = training_sequence_text.replace('\r', ' ').replace('\n', ' ')

    # Clean the text to match the alphabet (convert to lowercase, remove punctuation)
    training_sequence = training_sequence_text.lower().replace('.', '').replace(',', '')

    # Build the vocabulary from all unique characters in the training data
    alphabet = ''.join(sorted(set(training_sequence)))
    vocab_size = len(alphabet)
    MAX_LEN = 8  # Define the maximum sequence length for padding

    # We will use a sliding window of size MAX_LEN as input features (X)
    # and the next character as the target label (y).
    X_train_list = []
    y_train_list = []

    for i in range(len(training_sequence) - MAX_LEN):
        # Get the input sequence and pad it
        input_chars = training_sequence[i:i+MAX_LEN]
        one_hot_input = pad_sequence(input_chars, MAX_LEN, alphabet)

        # Get the target character and one-hot encode it
        target_char = training_sequence[i+MAX_LEN]
        one_hot_target = char_to_one_hot(target_char, alphabet)

        X_train_list.append(one_hot_input)
        y_train_list.append(one_hot_target)

    # Convert the lists to NumPy arrays for efficient computation
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # 2. Train the neural network
    # Input size is MAX_LEN * vocab_size
    # Output size is vocab_size
    input_size = MAX_LEN * vocab_size
    output_size = vocab_size
    # Hidden layer sizes for the three layers
    hidden_size_1 = 32
    hidden_size_2 = 32
    hidden_size_3 = 32

    model = SimpleNeuralNet(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, hidden_size_3=hidden_size_3, output_size=output_size, dropout_rate=0.2)
    # Try to load weights if available
    model.load_weights("model_weights.npz")
    # Train the model over many epochs
    model.train(X_train, y_train, epochs=100000, learning_rate=0.01)
    
    # 3. Test predictions on the training data itself
    print("\n--- Testing Predictions on Training Data ---")
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(training_sequence) - MAX_LEN):
        input_chars = training_sequence[i:i+MAX_LEN]
        actual_next_char = training_sequence[i+MAX_LEN]
        
        one_hot_input = pad_sequence(input_chars, MAX_LEN, alphabet)
        input_for_prediction = np.array([one_hot_input])
        
        prediction_vector = model.predict(input_for_prediction)
        predicted_char = one_hot_to_char(prediction_vector[0], alphabet)
        
        is_correct = "✓" if predicted_char == actual_next_char else "✗"
        if predicted_char == actual_next_char:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"Input: '{input_chars}' -> Actual next: '{actual_next_char}', Predicted: '{predicted_char}' {is_correct}")

    # Display the final accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nPrediction accuracy on training data: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")
    
    # 4. Generate sequences of 32 characters
    print("\n--- Generating Sequences of characters ---")
    # Use the first four unique words from the training data as prompts
    words = []
    for word in training_sequence.split():
        if word not in words:
            words.append(word)
        if len(words) == 10:
            break
    start_sequences = words

    for start_seq in start_sequences:
        generated_text = generate_sequence(model, start_seq, 32, MAX_LEN, alphabet)
        print(f"Prompt: '{start_seq}' -> Generated: '{generated_text}'")


