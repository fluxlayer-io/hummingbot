import os
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

from hummingbot.fluxlayer_api.rfq import get_generic_rfq_request

app_host = "0.0.0.0"
app_port = 8081

class RFQRequest(BaseModel):
    source_chain: str = Field(..., description="源链")
    source_token: str = Field(..., description="源token")
    amount: float = Field(..., description="交易数量")
    target_chain: str = Field(..., description="目标链")
    target_token: str = Field(..., description="目标token")
    is_buy: bool = Field(..., description="是否买入")

server = FastAPI()

@server.post("/rfq_request")
async def get_rfq_request(req: RFQRequest):
    data = await get_generic_rfq_request(
        req.source_chain, req.source_token, req.amount,
        req.target_chain, req.target_token, req.is_buy
    )
    return data

if __name__ == "__main__":
    uvicorn.run(server, host=app_host, port=app_port)