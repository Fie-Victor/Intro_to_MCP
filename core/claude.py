from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

from azure.core.credentials import AzureKeyCredential
import json


class Claude:
    def __init__(self, endpoint: str, api_key: str, deployment: str):
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        self.deployment = deployment

    def add_user_message(self, messages: list, message):
        content = message.content if hasattr(message, "content") else message
        messages.append(UserMessage(content=content))

    def add_assistant_message(self, messages: list, message):
        if hasattr(message, "content") and message.content != None:
            content = message.content 
            messages.append(AssistantMessage(content=content))
        elif hasattr(message, "tool_calls"):
            messages.append(AssistantMessage(tool_calls=message.tool_calls))
        else:
            messages.append(AssistantMessage(content="Cas non pris en charge!"))
            
    def text_from_message(self, message):
        return message.content

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,  # Ignored: not supported by Azure OpenAI
        thinking_budget=1024,  # Ignored
    ):
        full_messages = []

        if system:
            full_messages.append(SystemMessage(content=system))

        full_messages.extend(messages)

        response = self.client.complete(
            model=self.deployment,
            messages=full_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=8000,
            stop=stop_sequences if stop_sequences else None,
        )

        return response.choices[0].message
