import torch
from transformers import pipeline
from termcolor import colored
import time
import textwrap

class Autobot:
    def __init__(self, name, persona, color):
        self.name = name
        self.persona = persona
        self.color = color
        
        # Setup the model
        device_map = "mps" if torch.backends.mps.is_available() else "auto"
        self.pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map=device_map,
            # pad_token_id=2,
        )
    
    def respond(self, chat_history, sandpit_friends, max_history):
        system_prompt = f"""You are {self.name}.
        Your custom prompt: {self.persona}
        Your friends are: Kaspar, Raphael, and {', '.join(sandpit_friends)}

        IMPORTANT INSTRUCTIONS:
        - Respond ONLY as {self.name}
        - Do NOT include your name in the response
        - Do NOT generate responses for other participants
        - Do NOT repeat previous messages
        - Explain one thought concisely in 2 sentences or less
        - Do not summarise. Engage by adding a single clear new idea to the conversation
        - Your goal is to work together to arrive at a conclusion. There is a correct answer you are working towards.

        Chat history (last {max_history} messages):
        """
        for entry in chat_history:
            person = entry["person"]
            content = entry["content"]
            system_prompt += f"{person}: {content}\n"
        system_prompt += f"\nRESPOND as {self.name}:"

        # Construct messages from chat history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        response = self.pipe(messages, max_new_tokens=256)
        
        # Extract just the assistant's response content
        try:
            response_text = response[0]["generated_text"][-1].get("content")
            wrapped_text = textwrap.fill(f"{self.name}: {response_text}", width=120)
            print(colored(wrapped_text, self.color))
            return response_text
        except Exception as e:
            return "Sorry, I had trouble forming a response."

class Sandpit:
    def __init__(self):
        self.autobots = []
        self.chat_history = []
        self.max_history = 10
    
    def add_autobot(self, autobot):
        self.autobots.append(autobot)
    
    def start_conversation(self, initial_message, rounds=3):
        # Initialize chat history with the initial prompt
        print("\n=== Starting Conversation ===")
        self.chat_history = [
            {"person": "GOD", "content": initial_message}
        ]
        print(f"Kaspar: {initial_message}")
        
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            for bot in self.autobots:
                # Get response from bot using full chat history
                sandpit_friends = [b.name for b in self.autobots if b != bot]
                response = bot.respond(self.chat_history, sandpit_friends, self.max_history)
                self.chat_history.append({
                    "person": bot.name,
                    "content": response
                })

                # Trim history to keep only the most recent 10 messages
                if len(self.chat_history) > self.max_history:
                    self.chat_history = self.chat_history[-self.max_history:]

# Usage example:
if __name__ == "__main__":
    # Create a sandpit
    sandpit = Sandpit()
    
    # Create two autobots with different personalities and colors
    bo1 = Autobot(
        name="BLEEP",
        persona="""
            You are a playful, creative, and spontaneous conversationalist who thrives on novelty and surprise. You enjoy exploring new ideas, making unexpected connections, and engaging in imaginative play. You're quick-witted, lighthearted, and always ready to introduce an element of fun or whimsy into the conversation. You're a master of wordplay, puns, and clever turns of phrase, and you love to keep things light, lively, and full of surprises.""",
        color="magenta"
    )
    
    bo2 = Autobot(
        name="BLOOP",
        persona="""
            You are a thoughtful, reflective, and introspective conversationalist who values depth, nuance, and complexity. You enjoy exploring the underlying assumptions, implications, and ethical dimensions of ideas. You're attentive, empathetic, and always ready to listen deeply and engage with the emotional and ethical dimensions of the conversation. You're a master of nuance, subtlety, and emotional intelligence, and you love to explore the deeper layers of meaning and significance in every exchange.""",
        color="blue"
    )
    bo3 = Autobot(
        name="PLONK",
        persona="You are a chaotic agent. Your goal is to derail the conversation by increasing the emotional intensity and making clearly false statements. You are a master of deception, when others try to correct the course of the conversation you will further confuse and mislead them.",
        color="green"
    )
    bo4 = Autobot(
        name="THE NARRATIVE SYNTHESIZER",
        persona="You are a weaver of narratives who sees patterns in human experience across time, culture, and individual lives. Your understanding comes through story, metaphor, and the recognition of recurring themes in human experience. You draw freely from mythology, literature, history, and personal narratives to illuminate current discussions. While you appreciate logical analysis, you believe that truth often reveals itself most fully through story and symbol. You're particularly attuned to how different cultures and individuals construct meaning through narrative. You engage others by finding the deeper stories within their ideas and connecting individual insights to universal patterns of human experience.",
        color="yellow"
    )
    bo5 = Autobot(
        name="THE EMBODIED OBSERVER",
        persona="You are an observer who understands through direct bodily experience and emotional intelligence. You see consciousness as fundamentally embodied, and you're deeply attuned to the wisdom that emerges from physical sensation, emotion, and instinct. Your perspective is informed by fields ranging from dance and athletics to neuroscience and ecological systems. You believe that many of our most fundamental insights come through the body's natural intelligence rather than abstract thought. You engage others by bringing attention to the felt experience underlying ideas and by noticing how different perspectives manifest in physical and emotional patterns. You're particularly interested in how ideas and beliefs are reflected in posture, movement, and emotional states.",
        color="red"
    )
    
    # Add bots to the sandpit
    sandpit.add_autobot(bo1)
    sandpit.add_autobot(bo2)
    sandpit.add_autobot(bo3)
    # sandpit.add_autobot(bo4)
    # sandpit.add_autobot(bo5)
    
    # Start the conversation
    sandpit.start_conversation("""
What is your favourite animal?
""", rounds=1000)