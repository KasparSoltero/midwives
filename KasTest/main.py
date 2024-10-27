import os
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.formatted_text import HTML
import json
from datetime import datetime
from typing import Optional, Dict, List, Union
from rich.theme import Theme
from rich.console import Console
from rich.markdown import Markdown
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv

import anthropic
import openai

# Load and immediately verify all env contents
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

class ChatInterface:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.console = Console(theme=Theme({
            "user": "green bold",
            "assistant": "cyan bold",
            "system": "red italic",
        }))
        
        # Prompt toolkit style
        self.prompt_style = PromptStyle.from_dict({
            'user': '#00ff00 bold',
            'assistant': '#00ffff bold',
            'system': '#ff0000 italic',
        })
        
        self.session = PromptSession()
        self.conversation_history: List[Dict] = []
        self.current_model = "claude-3-opus-20240229"
        self.system_prompt = "You are a helpful AI assistant."
        self.temperature = 0.7

    async def send_message_anthropic(self, message: str) -> str:
        response = self.anthropic_client.messages.create(
            model=self.current_model,
            messages=[{"role": "user", "content": message}],
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.content[0].text

    async def send_message_openai(self, message: str) -> str:
        messages = [{"role": "user", "content": message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
            
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def save_history(self, filename: str = "chat_history.json"):
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f)

    def load_history(self, filename: str = "chat_history.json"):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)

    def handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        if cmd == "/exit":
            return False
        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif cmd.startswith("/model "):
            self.current_model = cmd.split(" ")[1]
        elif cmd.startswith("/temp "):
            self.temperature = float(cmd.split(" ")[1])
        elif cmd.startswith("/system "):
            self.system_prompt = " ".join(cmd.split(" ")[1:])
        elif cmd == "/save":
            self.save_history()
        elif cmd == "/load":
            self.load_history()
        return True

    def format_message(self, role: str, content: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] [{role}]{content}[/{role}]"

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        while True:
            try:
                # Multi-line input support
                user_input = await self.session.prompt_async(
                    HTML('<user>You: </user>'),
                    style=self.prompt_style,
                    multiline=True,
                )

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                self.conversation_history.append({"role": "user", "content": user_input})
                self.console.print(self.format_message("user", user_input))

                if self.current_model.startswith("claude"):
                    response = await self.send_message_anthropic(user_input)
                else:
                    response = await self.send_message_openai(user_input)

                self.conversation_history.append({"role": "assistant", "content": response})
                self.console.print(self.format_message("assistant", response))

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print("Error:", style="red bold")
                self.console.print(traceback.format_exc(), style="red")

def main():
    chat = ChatInterface()
    chat.run()

if __name__ == "__main__":
    main()