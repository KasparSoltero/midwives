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
import time
import random
import threading

from transformers import pipeline
import torch

import anthropic
import openai

# Load and immediately verify all env contents
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

class ChatInterface:
    def __init__(self):
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
        self.current_model = "meta-llama/Meta-Llama-3-8B"#"meta-llama/Llama-3.2-1B-Instruct"#"claude-3-5-sonnet-20241022"#"claude-3-opus-20240229"
        self.system_prompt = ""
        self.temperature = 0.7
        self.loop_running = False

    async def send_message_llama(self, message: str, end="\n") -> str:
        # Setup the model if not already created
        if not hasattr(self, 'llama_model'):
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            
            device_map = "mps" if torch.backends.mps.is_available() else "auto"
            print(f'llama device_map: {device_map}')
            self.llama_tokenizer = AutoTokenizer.from_pretrained(self.current_model)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.current_model,
                torch_dtype=torch.float16,
                device_map=device_map
            )
            self.streamer = TextIteratorStreamer(self.llama_tokenizer)
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Prepare input with proper device placement
        inputs = self.llama_tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        )

        # Move input tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Setup generation in a separate thread
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "streamer": self.streamer,
            "max_new_tokens": 300,
            "temperature": self.temperature,
            "do_sample": True,
            "pad_token_id": self.llama_tokenizer.pad_token_id or self.llama_tokenizer.eos_token_id,
        }
        
        thread = threading.Thread(target=self.llama_model.generate, kwargs=generation_kwargs)
        thread.start()

        # Collect and display the streamed output
        tokens_to_remove = ['<|begin_of_text|>', '<|eot_id|>', '<|end_of_text|>', '|<|eom_id|>', '<|start_header_id|>', ]
        collected_content = []
        message_length = len(message)
        accumulated_text = ""

        for text in self.streamer:
            # Clean the text before displaying
            cleaned_text = text
            for token in tokens_to_remove:
                cleaned_text = cleaned_text.replace(token, '')
                
            if cleaned_text.strip():  # Only process if there's content after cleaning
                collected_content.append(cleaned_text)
                accumulated_text += cleaned_text
                
                # Only start printing after we've accumulated more than the original message
                if len(accumulated_text) > message_length:
                    if not hasattr(self, 'started_printing'):
                        self.started_printing = True
                        # Print only the part after the message
                        self.console.print(accumulated_text[message_length:], end="", style="assistant")
                    else:
                        self.console.print(cleaned_text, end="", style="assistant")
                
        thread.join()
        full_response = "".join(collected_content)
        if full_response.startswith(message):
            full_response = full_response[len(message):].lstrip()

        self.console.print(end, end="")
        delattr(self, 'started_printing') if hasattr(self, 'started_printing') else None
        return full_response

    async def send_message_anthropic(self, message: str) -> str:
        if not hasattr(self, 'anthropic_client'):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        response = self.anthropic_client.messages.create(
            model=self.current_model,
            messages=[{"role": "user", "content": message}],
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=1024,
            stream=True,
        )
        print(f'')
        # return response.content[0].text
        collected_content = []
        for chunk in response:
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                chunk_text = chunk.delta.text
                collected_content.append(chunk_text)
                self.console.print(chunk_text, end="", style="assistant")
                # chunk_text = chunk.delta.text
                # collected_content.append(chunk_text)
                # self.console.print(chunk_text, end="", style="assistant")
        
        full_response = "".join(collected_content)
        self.console.print()  # New line after streaming completes
        return full_response

    async def send_message_openai(self, message: str) -> str:
        if not hasattr(self, 'openai_client'):
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        messages = [{"role": "user", "content": message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
            
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        # return response.choices[0].message.content
        collected_content = []
        async for chunk in response:
            if chunk.choices:
                chunk_text = chunk.choices[0].message.content
                collected_content.append(chunk_text)
                self.console.print(chunk_text, end="", style="assistant")

        full_response = "".join(collected_content)
        self.console.print()  # New line after streaming completes
        return full_response

    async def continuous_generation(self, initial_prompt: str, context_window: int = 200):
        self.loop_running = True
        try:
            tokens_to_remove = ['<|begin_of_text|>', '<|eot_id|>', '<|end_of_text|>', '<|eom_id|>', '<|start_header_id|>', ]
            context = initial_prompt
            print(f'...{initial_prompt}.')
            while self.loop_running:
                # print(f'\nsending response:\n\n{context}\n\n')
                response = await self.send_message_llama(context, end="")
                # print(f'\nresponse:\n\n{response}\n\n')
                # remove tokens
                for token in tokens_to_remove:
                    response = response.replace(token, '')
                # print the response
                context = (context + response)
                if len(context) > context_window:
                    context = context[-context_window:]
                await asyncio.sleep(1)
        except Exception as e:
            print(f"\nError in continuous generation: {e}")
        finally:
            self.loop_running = False

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
        elif cmd == "/clear-terminal":
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

        elif cmd == "/loop":
            # generate one random letter
            random_letter = chr(random.randint(97, 122))
            asyncio.create_task(self.continuous_generation(
                initial_prompt=random_letter,
            ))

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

                timestamp = datetime.now().strftime("%H:%M:%S")
                self.console.print(f"[{timestamp}]", end=" ", style="assistant")
                self.console.print(f"[{self.current_model}]", end=" ", style="assistant")

                if self.current_model.startswith("claude"):
                    response = await self.send_message_anthropic(user_input)
                elif self.current_model.startswith("meta-llama"):
                    response = await self.send_message_llama(user_input)
                elif self.current_model.startswith("gpt"):
                    response = await self.send_message_openai(user_input)
                else:
                    raise ValueError(f"Unknown model: {self.current_model}")

                self.conversation_history.append({"role": "assistant", "content": response})


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