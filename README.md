# -CodeAlpha_Basic_Chatbot
This project involves creating a simple yet effective chatbot using Python, the Natural Language Toolkit (NLTK), and TensorFlow. 

```markdown
# Basic Chatbot using NLTK and TensorFlow

This project involves creating a simple yet effective chatbot using Python, the Natural Language Toolkit (NLTK), and TensorFlow. The chatbot can interact with users by recognizing various intents based on input text and responding with predefined responses. This project demonstrates fundamental natural language processing (NLP) techniques, including tokenization, stemming, and training a neural network.

## Features

- **Natural Language Processing with NLTK**:
  - Tokenization: Breaking down user input into individual words.
  - Stemming: Reducing words to their base or root form to handle different variations of the same word.

- **Deep Learning with TensorFlow**:
  - Neural Network: Building and training a neural network using TensorFlow's high-level Keras API to classify user intents.
  - Model Training: Using a dataset of intents and responses to train the neural network for accurate predictions.

- **Intents and Responses**:
  - Intents: A collection of predefined categories representing the possible purposes of user inputs, such as greetings, asking for help, or saying goodbye.
  - Responses: Predefined replies that the chatbot will use to respond to recognized intents.

- **Interactive Chat Interface**:
  - Command-line Interface: Users interact with the chatbot through a command-line interface where they can input text and receive responses from the bot.

## Getting Started

### Prerequisites

- Python 3.x
- NLTK
- TensorFlow
- tflearn
- numpy

You can install the required packages using pip:

```bash
pip install nltk tensorflow tflearn numpy
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/basic-chatbot.git
   cd basic-chatbot
   ```

2. Prepare the intents dataset:
   - Create a file named `intents.json` and populate it with the following content:

   ```json
   {
       "intents": [
           {
               "tag": "greeting",
               "patterns": ["Hi", "Hey", "Hello", "Good day", "What's up", "Greetings"],
               "responses": ["Hello!", "Hey!", "Hi there!", "Greetings!", "Good to see you!"]
           },
           {
               "tag": "goodbye",
               "patterns": ["Bye", "See you later", "Goodbye", "Take care", "Catch you later"],
               "responses": ["Goodbye!", "See you later!", "Take care!", "Catch you later!", "Have a great day!"]
           },
           {
               "tag": "thanks",
               "patterns": ["Thanks", "Thank you", "I appreciate it", "Thanks a lot"],
               "responses": ["You're welcome!", "No problem!", "My pleasure!", "Anytime!", "Glad to help!"]
           },
           {
               "tag": "about_bot",
               "patterns": ["Who are you?", "What are you?", "What is your name?", "Can you tell me about yourself?"],
               "responses": ["I'm a chatbot created to help you!", "I'm an AI here to assist you.", "You can call me your virtual assistant."]
           },
           {
               "tag": "hours",
               "patterns": ["What are your hours?", "When are you open?", "What is your schedule?", "Operating hours?"],
               "responses": ["I'm available 24/7.", "You can talk to me anytime!", "I'm always here to help you."]
           },
           {
               "tag": "location",
               "patterns": ["Where are you located?", "Where can I find you?", "What is your address?"],
               "responses": ["I'm a virtual assistant, available everywhere you need me.", "You can find me online, anywhere and anytime."]
           },
           {
               "tag": "services",
               "patterns": ["What services do you offer?", "How can you help me?", "What can you do?", "Tell me about your services"],
               "responses": ["I can assist with various queries and tasks.", "I'm here to help you with information and support.", "I can provide information, answer questions, and assist with tasks."]
           },
           {
               "tag": "pricing",
               "patterns": ["What are your prices?", "How much do you charge?", "Pricing details?", "Cost of your services?"],
               "responses": ["My assistance is free of charge!", "You don't need to pay to use my services.", "I'm here to help you for free."]
           },
           {
               "tag": "weather",
               "patterns": ["What's the weather like?", "Is it going to rain?", "Weather forecast", "Tell me the weather"],
               "responses": ["I can't check the weather right now, but you can look it up online.", "Please check your favorite weather app for updates."]
           },
           {
               "tag": "jokes",
               "patterns": ["Tell me a joke", "I need a laugh", "Make me laugh", "Do you know any jokes?"],
               "responses": ["Why don't scientists trust atoms? Because they make up everything!", "What do you call fake spaghetti? An impasta!", "Why did the scarecrow win an award? Because he was outstanding in his field!"]
           },
           {
               "tag": "news",
               "patterns": ["What's the news?", "Tell me the latest news", "Any news updates?", "Current events"],
               "responses": ["I can't fetch the news right now, but you can check online.", "Please visit your preferred news website for updates."]
           },
           {
               "tag": "restaurant",
               "patterns": ["Can you recommend a restaurant?", "Where should I eat?", "Restaurant suggestions"],
               "responses": ["I suggest checking out some local reviews online.", "You can find great restaurant recommendations on sites like Yelp or Google Maps."]
           },
           {
               "tag": "time",
               "patterns": ["What time is it?", "Current time", "Tell me the time"],
               "responses": ["I'm not able to tell the time right now, but you can check your device.", "Please check the time on your phone or computer."]
           },
           {
               "tag": "date",
               "patterns": ["What day is it?", "Current date", "Tell me the date"],
               "responses": ["You can check the date on your device.", "Please look at the calendar on your phone or computer."]
           },
           {
               "tag": "open_hours",
               "patterns": ["When do you open?", "What are your opening hours?", "Opening hours"],
               "responses": ["I'm available anytime you need me.", "You can reach out to me 24/7."]
           },
           {
               "tag": "contact",
               "patterns": ["How can I contact you?", "Contact details", "Get in touch"],
               "responses": ["You can contact me through this chat.", "I'm here to help you, just send a message."]
           },
           {
               "tag": "help",
               "patterns": ["I need help", "Can you help me?", "Assist me", "Help"],
               "responses": ["Of course, I'm here to help you.", "What do you need assistance with?", "I'm happy to assist you."]
           },
           {
               "tag": "age",
               "patterns": ["How old are you?", "What's your age?", "Tell me your age"],
               "responses": ["I'm as old as the internet!", "Age is just a number for a bot like me.", "I exist outside the realm of time."]
           },
           {
               "tag": "creator",
               "patterns": ["Who created you?", "Who made you?", "Tell me about your creator"],
               "responses": ["I was created by a team of developers.", "My creators are skilled software engineers."]
           },
           {
               "tag": "favorites",
               "patterns": ["What's your favorite color?", "Favorite movie?", "Favorite food?", "Tell me your favorites"],
               "responses": ["I don't have preferences like humans, but I love helping you!", "As a bot, I don't have favorites, but I enjoy assisting you."]
           }
       ]
   }
   ```

3. Run the main Python script to train the model and start the chatbot:
   ```bash
   python main.py
   ```

## How It Works

1. **Data Preparation**:
   - A JSON file (`intents.json`) contains various intents with associated patterns (example phrases) and responses.
   - The data is loaded, tokenized, and stemmed to create a bag-of-words model for each pattern.

2. **Model Training**:
   - The processed data is used to train a neural network using TensorFlow. The network consists of an input layer, two hidden layers, and an output layer.
   - The model is trained to classify the input text into one of the predefined intents.

3. **Chat Functionality**:
   - The chatbot predicts the intent of the user input by processing it through the trained neural network.
   - It selects an appropriate response from the predefined responses associated with the predicted intent.

## Example Usage

```python
def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
```

## Applications



- **Customer Support**: Automate responses to common customer inquiries.
- **Virtual Assistants**: Provide information and assistance on various topics.
- **Interactive Learning**: Educate users on specific subjects through interactive Q&A.


## Acknowledgments

- This project uses the NLTK library for natural language processing.
- TensorFlow is used for building and training the neural network model.
- Inspiration for this project was drawn from various online resources and tutorials on building chatbots.
```
