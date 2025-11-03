
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.prebuilt import create_react_agent

from Prompts.specialized_agent_prompt import (
    get_supervisor_prompt,
    get_claudeFlow_agent_prompt,
)
from Tools.claudeFlow_agent_tool import call_claude_flow_mle_star



# =========================  Model Config  ===================================
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1", 
    api_key="ollama",                      
    model="qwen2.5:7b-instruct",         
    temperature=0,
    parallel_tool_calls = False
)


def call_langGraph_agents(user_query: str):

    supervisor_node = create_agent("supervisor")
    claudeFlow_agent_node = create_agent("claudeFlow_agent")

    multi_conversation = create_swarm(
        agents=[
            supervisor_node,
            claudeFlow_agent_node,
        ],
        default_active_agent="supervisor",
    )

    app = multi_conversation.compile()
    response = app.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
    )
    print("="* 40, "Start of the Messages in the Conversation", "="* 40)
    for m in response["messages"]:
        print(m.pretty_print())
    print("="* 40, "End of the Messages in the Conversation", "="* 40)


    final_response = response["messages"][-1].content
    print("="* 80)
    print("\nFinal response:", final_response)
    print("="* 80)

    return final_response


def create_agent(agent_name: str):

    match agent_name:

        case "supervisor":
            supervisor_prompt = get_supervisor_prompt()
            supervisor = create_react_agent(
                model=llm,
                tools=[
                    create_handoff_tool(agent_name="claudeFlow_agent"),
                ],
                name="supervisor",
                prompt=supervisor_prompt,
            )
            return supervisor

        case "claudeFlow_agent":
            claudeFlow_agent_prompt = get_claudeFlow_agent_prompt()
            claudeFlow_agent = create_react_agent(
                model=llm,
                tools=[call_claude_flow_mle_star,
                       create_handoff_tool(agent_name="supervisor")],
                name="claudeFlow_agent",
                prompt=claudeFlow_agent_prompt,
            )
            return claudeFlow_agent

    raise ValueError(f"Unknown agent name: {agent_name}")
