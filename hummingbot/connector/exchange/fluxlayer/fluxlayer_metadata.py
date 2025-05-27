from typing import Dict

from pydantic import BaseModel


class TradingPairMetadata(BaseModel):
    source_chain: str
    source_token: str
    target_chain: str
    target_token: str


class FluxLayerMetadata(BaseModel):
    api_endpoint: str
    amount: str
    trading_pairs: Dict[str, TradingPairMetadata]
