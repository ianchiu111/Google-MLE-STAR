from graph import call_langGraph_agents

User_Query = "please use the data/train.csv dataset and Sales as target column to predict sales revenue."
# User_Query = "Yes, I want to execute with previous information."

if __name__ == "__main__":
    final_output = call_langGraph_agents(user_query=User_Query)
    print("Final Output from LangGraph Agents:\n", final_output)

