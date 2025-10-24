import os
import json
import numpy as np
import pandas as pd


from typing import Annotated, Literal, List, Union
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from Langgraph.Prompts.keywords_agent_tool_prompt import (
    get_customer_list_agent_prompt,
    get_keywords2keywords_agent_prompt,
    get_result_agent_prompt,
)

from Langgraph.utils.AI_SaaS_utils.azure_openai_helper import get_azure_openai_instance
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
load_dotenv(".env")


# =========================  Model Config  ===================================

unparallel_model = get_azure_openai_instance(
    deployment_name="gpt-4o", temperature=0, parallel_tool_calling=False
)

model = get_azure_openai_instance(
    deployment_name="gpt-4o",
    temperature=0,
)

# ========================== Tools ===================================

@tool
def md_to_industry(
    model_name: Annotated[Union[List[str], str], " key related to 'products' "]= None,
):
    """
    Input: model name 
    Output: industry category and customer information
    """

    if not model_name or model_name == "空" or model_name == "None":
        print("使用者無輸入 model name 因此不進行客戶產業查詢")
        return "industry categories is empty."

    like_clauses = " OR ".join([f"part = '{p}'" for p in model_name])

    now_year = datetime.now().year
    one_year_ago = now_year - 1
    ##⚠️⚠️⚠️ limit 3 是 dataforseo api 限制的暫緩處理 
    sql_query = f"""
    SELECT 
        part,
        customername,
        SUM(us_amt) AS total_us_amt
    FROM (
        SELECT DISTINCT
            customername,
            us_amt,
            part,
            EXTRACT(YEAR FROM (CAST(ymd AS DATE))) AS year_id
        FROM dx_management.rv_eai_acldw_shipmentbacklog_bomexpand_fa_full
        WHERE region = 'Taiwan'
        AND EXTRACT(YEAR FROM (CAST(ymd AS DATE))) BETWEEN {one_year_ago} AND {now_year}
        AND {like_clauses}
    ) t
    GROUP BY part, customername
    ORDER BY total_us_amt DESC
    LIMIT 3
    """
    logger.info(f"\nSQL Query:\n {sql_query}")

    ### extract data from Denodo DB with SQL Query
    try:
        data = ai_saas_client.query_database(denodoserver_name, sql_query)

    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({"message": f"Error: {e}"}, ensure_ascii=False)

    df = pd.DataFrame(data['data'])

    if df.empty:
        logger.error("Error: Data is empty after executing SQL Query on DenodoDB.")
        # return json.dumps({"message": f"Error: Data is empty after executing SQL Query on DenodoDB."}, ensure_ascii=False)

    else:
        name_list = df.customername.unique().tolist()

        all_industries = "目標客戶的產業類別有: "

        ## 非同步 customer2industry function
        with ThreadPoolExecutor(max_workers=4) as customer2industry_executor:

            LLMResponse = {customer2industry_executor.submit(customer2industry, c): c for c in name_list}
            for response in as_completed(LLMResponse):

                try:
                    customer, industry_category = response.result()
                    if industry_category is None:
                        all_industries += f"\n客戶名稱: {customer}\n客戶相關資訊：<取得失敗>\n"
                    else:
                        all_industries += f"\n客戶名稱: {customer}\n客戶相關資訊：{industry_category}\n"
                except Exception as e:
                    print(f"❌Error: {e}")

        print("✅✅✅", all_industries)
        return all_industries


@tool
def prepare_keyword_list(
    industry_categories: Annotated[List[str], "Industry category in string list format with multiple industry categories."],
    requirements: Annotated[int, "User's Requirements"],
):
    """
    use specific csv file to print out all the keywords
    1. 目前只有整理**產關鍵字**的回答
    """
    print("== 所有產業 ==\n", industry_categories)
    if not industry_categories:
        industry_categories = ["未使用特定產業類別"]
    
    task_id = get_task_id()
    all_body = []
    for industry_category in industry_categories:
        # 讀取對應的 CSV 檔案
        if not industry_category:
            industry_category = "無產業類別"

        csv_file_path = f"Langgraph/resultByIndustryCategory/{industry_category}_biddingKeywords.csv"

        print(f"✴️ industry_category: {industry_category}")
        print(f"✴️ csv_file_path: {csv_file_path}")
        print(f"✴️ requirements: {requirements}")

        # try:
        df = pd.read_csv(csv_file_path)
        if df.empty or 'keyword' not in df.columns:
            return json.dumps({"error": f"No data or missing 'keyword' in {csv_file_path}."}, ensure_ascii=False)
        
        # 過濾含品牌黑名單關鍵字
        ## ⚠️⚠️⚠️ 現在是 txt 寫死去過濾，之後要動態篩選品牌名稱
        brand_blacklist = load_brand_blacklist('brand_blacklist.txt')
        df = df[~df['keyword'].apply(lambda x: is_brand_in_keyword(x, brand_blacklist))]

        # 按cpc排序，取前requirements個
        # filtered = df[df['keyword'].apply(contains_chinese)].copy()
        # 測試把篩選中文拿掉看結果如何?
        filtered = df.copy()

        filtered['cpc'] = filtered['cpc'].fillna(99.99)
        
        filtered_sorted = filtered.sort_values(
            by=['cpc'],
            ascending=[True]
        ).reset_index(drop=True)
        filtered_sorted = filtered_sorted.head(requirements)

        filtered_keywords = filtered_sorted[['domain','keyword', 'cpc', 'competition_index']]
        # filtered_keywords_list = filtered_keywords.to_dict('records')
        # results[industry_category] = filtered_keywords_list

        # prepare_adaptive_card
        columns = filtered_keywords.columns.tolist()
        data_values = filtered_keywords.values.tolist()

        # 表頭 + 資料
        # 1. 增加產業小標題
        all_body.append({
            "type": "TextBlock",
            "text": f"【{industry_category}】",
            "weight": "bolder",
            "size": "large",
            "wrap": True,
            "separator": True
        })

        # 2. Header
        header_columns = []
        for col_name in columns:
            header_columns.append({
                "type": "Column",
                "width": "stretch",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": col_name,
                        "weight": "bolder",
                        "wrap": True
                    }
                ]
            })
        all_body.append({
            "type": "ColumnSet",
            "columns": header_columns
        })

        # 3. Data
        for row in data_values:
            row_columns = []
            for cell_value in row:
                row_columns.append({
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": str(cell_value),
                            "wrap": True
                        }
                    ]
                })
            all_body.append({
                "type": "ColumnSet",
                "columns": row_columns
            })

    # ======最後寫成一個大 adaptive card======
    adaptive_card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.0",
        "body": all_body
    }

    file_path = f"Langgraph/result_for_adaptiveCard/{task_id}.json"
    with open(file_path , 'w', encoding='utf-8') as json_file:
        json.dump(adaptive_card, json_file, ensure_ascii=False, indent=4)

    print("✅ Adaptive card json file saved. ✅")
    return "Adptive card json file saved."
            # print("✴️✴️✴️ Final Results:", results)
            # return results

        # except Exception as e:
        #     print(f"Failed to process {csv_file_path}: {e}")
        #     return json.dumps({"error": f"Failed to process {csv_file_path}: {e}."}, ensure_ascii=False)
    


# ================================================================


def create_graph():
    checkpoint = InMemorySaver()

    customer_list_agent_node = create_agent("customer_list_agent")
    keywords2keywords_agent_node = create_agent("keywords2keywords_agent")
    result_agent_node = create_agent("result_agent")

    workflow = StateGraph(MessagesState)  # GraphState
    workflow.add_edge(START, "customer_list_agent")
    workflow.add_node("customer_list_agent", customer_list_agent_node)   
    workflow.add_node("keywords2keywords_agent", keywords2keywords_agent_node)
    workflow.add_node("result_agent", result_agent_node)

    workflow.add_edge("customer_list_agent", "keywords2keywords_agent")
    workflow.add_edge("keywords2keywords_agent", "result_agent")
    workflow.add_edge("result_agent", END)

    graph = workflow.compile(checkpointer=checkpoint)
    return graph


@tool
def keyword_agent_process_tool():
    """
    Prefer Query_Dict when provided. 
    Fallback to user_query only if bundle is missing.
    """


    graph = create_graph()

    response = graph.invoke(
        {"messages": clean_query}, 
    )

    print("")
    for m in response["messages"]:
        print(m.pretty_print())

    final_response = response["messages"][-1].content

    return final_response


def create_agent(agent_name: str):
    match agent_name:

        case "customer_list_agent":

            customer_list_agent_prompt = get_customer_list_agent_prompt()
            customer_list_agent = create_react_agent(
                unparallel_model, 
                tools=[md_to_industry], 
                prompt=customer_list_agent_prompt
            )

            def customer_list_agent_node(state: MessagesState) -> Command[Literal["keywords2keywords_agent"]]:
                
                logger.info(f"\n==========🔄️ customer_list_agent is processing 🔄️==========\n")
                result = customer_list_agent.invoke(state)   

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="customer_list_agent")
                        ]
                    },
                    goto="keywords2keywords_agent",
                )
            return customer_list_agent_node

        case "keywords2keywords_agent":

            keywords2keywords_agent_prompt = get_keywords2keywords_agent_prompt()
            keywords2keywords_agent = create_react_agent(
                unparallel_model, 
                tools=[keywords2keywords], 
                prompt=keywords2keywords_agent_prompt
            )

            def keywords2keywords_agent_node(state: MessagesState) -> Command[Literal["result_agent"]]:
                
                logger.info(f"\n==========🔄️ keywords2keywords_agent is processing 🔄️==========\n")
                print("✴️✴️✴️:", state["messages"][-1].content)
                result = keywords2keywords_agent.invoke(state)   

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="keywords2keywords_agent")
                        ]
                    },
                    goto="result_agent",
                )
            return keywords2keywords_agent_node

        case "result_agent":
            result_agent_prompt = get_result_agent_prompt()
            result_agent = create_react_agent(
                unparallel_model, 
                tools=[prepare_keyword_list], 
                prompt=result_agent_prompt
            )

            def result_agent_node(state: MessagesState) -> Command[Literal["__end__"]]:
                
                user_query = state["messages"][0].content
                category = state["messages"][-1].content
                clean_messages = "User's Original Query:" + user_query + "\nIndustry Category:" + category

                result = result_agent.invoke({"messages": [{
                                                    "role": "user",
                                                    "content": clean_messages
                                                }]},)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="result_agent")
                        ]
                    },
                    goto="__end__",
                )
            return result_agent_node
