from decimal import Decimal
from typing import Dict

from pydantic import BaseModel


class TradingPairMetadata(BaseModel):
    source_chain: str
    source_token: str
    target_chain: str
    target_token: str
    is_buy: bool
    source_amount: Decimal
    target_amount: Decimal

class FluxLayerMetadata(BaseModel):
    api_endpoint: str
    amount: str
    trading_pairs: Dict[str, TradingPairMetadata]
