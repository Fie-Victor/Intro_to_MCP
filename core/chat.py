from core.claude import Claude  # anciennement core.claude import Claude
from mcp_client import MCPClient
from core.tools import ToolManager
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage, ToolMessage



class Chat:
    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        self.claude_service: Claude = claude_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list = []

    async def _process_query(self, query: str):
        self.claude_service.add_user_message(self.messages, query)

    async def run(self, query: str) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            #print("Voici les messages au avant toolCall ou non Messages:", self.messages)
            
            response = self.claude_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.claude_service.add_assistant_message(self.messages, response)

            # Azure OpenAI n'utilise pas `stop_reason == "tool_use"` comme Anthropic
            # À adapter si tu fais du function/tool calling avec messages spéciaux
            if hasattr(response, "tool_calls") and response.tool_calls:
                #print(self.claude_service.text_from_message(response))

                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                # self.claude_service.add_user_message(
                #     self.messages, tool_result_parts
                # )
                self.messages.append(ToolMessage(tool_result_parts[0]["content"], tool_call_id=tool_result_parts[0]["tool_use_id"]))
            else:
                final_text_response = self.claude_service.text_from_message(
                    response
                )
                break

        return final_text_response
