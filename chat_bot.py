import random
import tensorflow as tf
import numpy as np


class BotAttributes:
    def __init__(self):
        self.awareness = 0  # Initialize awareness level to 0
        # Initialize conversation state dictionary with a "topic" key
        self.conversation_state = {"topic": None}
        self.max_tokens = 1000

        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.max_tokens, output_dim=64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.max_tokens, activation="softmax")
        ])

        # Compile the model
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")

    def generate_response(self, input_data):
        # Convert the input data to a string if it is a list
        if isinstance(input_data, list):
            input_data = " ".join(input_data)

        # Tokenize the input data
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.max_tokens, oov_token="<OOV>")
        tokenizer.fit_on_texts([input_data])
        input_sequences = tokenizer.texts_to_sequences([input_data])
        input_data = tf.keras.preprocessing.sequence.pad_sequences(
            input_sequences, padding="post")

        # Use the model to generate a response
        # Generate a probability distribution over the possible class labels
        response_probs = self.model.predict(input_data)[0]
        # Select the index of the maximum value in the probability distribution as the predicted class label
        response_class = np.argmax(response_probs)
        # Convert the predicted class label to text
        response = tokenizer.index_word[response_class]
        return response

    def maintain_conversation(self, input_data):
        # Use the conversation state to generate an appropriate response
        response = ""
        if "greeting" in self.conversation_state:
            response += "Hello again! "
        elif "goodbye" in self.conversation_state:
            response += "Goodbye! It was nice talking to you. "
            self.conversation_state.clear()
            return response

        # Update the conversation state based on the input data
        if any(word in input_data.lower() for word in ["hi", "hello", "hey"]):
            self.conversation_state["greeting"] = True
            response += "Hello! How are you doing today? "
        elif any(word in input_data.lower() for word in ["bye", "goodbye", "see you later"]):
            self.conversation_state["goodbye"] = True
            response += "Goodbye! It was nice talking to you. "
        else:
            response += self.generate_response(input_data)

        return response

    def generate_nonverbal(self):
        # Generate a random nonverbal response
        responses = ["*nods*", "*smiles*", "*frowns*"]
        return random.choice(responses)


ai = BotAttributes()

while True:
    try:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit the conversation
        if user_input.lower() in ["bye", "goodbye", "see you later"]:
            print("AI: " + ai.maintain_conversation(user_input))
            break

        # Send the input to the AI and receive a response
        ai_response = ai.maintain_conversation(user_input)
        print("AI: " + ai_response)

        # Generate a random nonverbal response
        nonverbal = ai.generate_nonverbal()
        print("AI " + nonverbal)
    except Exception as e:
        # If an exception is raised, print the error message and continue the loop
        print(e)
        continue

print("EXIT")
