
import os
import queue
from unittest import case
import pytz
import logging
import traceback
from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional
import time
import random
import string
import json
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from graph import call_langGraph_agents


app = Flask(__name__)
CORS(app)

taiwan_tz = pytz.timezone("Asia/Taipei")
current_time_taiwan = datetime.now(taiwan_tz)
version = current_time_taiwan.strftime("v%Y-%m%d-%H%M")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIResponse:
    """Static class for generating API responses"""

    @staticmethod
    def success(
        data: Any = None, message: str = "Success", meta: Optional[Dict] = None
    ):
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data is not None:
            response["data"] = data
        if meta:
            response["meta"] = meta
        return response

    @staticmethod
    def error(
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[str] = None,
    ):
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": error_code,
                "timestamp": datetime.now().isoformat(),
            },
        }
        if details:
            response["error"]["details"] = details
        return response, status_code


## ============================ Main API ============================
@app.route("/api/claude-flow/v1", methods=["POST"])
def processing_generation():
    """
    Analyze user query and generate keywords or content

    Body:
    {
        "query": "user's query string"
    }
    """

    try:
        if not request.is_json:
            return APIResponse.error(
                "The request must be in JSON format.", "INVALID_CONTENT_TYPE", 400
            )

        data = request.get_json()
        user_query = data.get("query")

        if not user_query:
            return APIResponse.error(
                "Missing required field 'query'.", "MISSING_REQUIRED_FIELD", 400
            )

        if not isinstance(user_query, str) or len(user_query.strip()) == 0:
            return APIResponse.error(
                "The query content cannot be empty.", "INVALID_QUERY", 400
            )

        # User_Query = "please use the data/train.csv dataset and Sales as target column to predict sales revenue."
        # User_Query = "Yes, I want to execute with previous information."

        print(user_query)

        final_output = call_langGraph_agents(user_query=user_query)
        print("Final Output from LangGraph Agents:\n", final_output)

        return final_output


    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

        error_response = {"done": True, "output": f"Processing failed: {str(e)}"}

        return jsonify(error_response), 500

@app.get("/health")
def health():
    return {"status": f"server is running {version}"}


if __name__ == "__main__":
    port = 5001  # Default port
    logger.info(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
