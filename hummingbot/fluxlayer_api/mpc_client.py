import os
from dataclasses import dataclass
from typing import Any, List

import requests


@dataclass
class MakerOrder:
    order_id: str
    cobo_id: str
    wallet_id: str
    source_chain: str
    target_chain: str
    i_token: str
    o_token: str
    iamount: str
    oamount: str
    slippage: str
    status: str
    fulfill_status: str
    maker_tx_hash: str
    sig: str
    created_at: str
    updated_at: str

@dataclass
class MPCAddress:
    chain_id: str
    wallet_id: str
    address: str
    encoding: str

class MPCClient:
    def __init__(self, api_host: str = None):
        self.api_host = api_host or os.environ.get("MPC_API_HOST")
        # self.api_host = "2kvAqkWZXcdoWXkVn3Ntrepsrp8jRBL9uJYRLLFU8DEY4jc1USkN8wiLoPFTLVMnaSQ9GfcuzZwShXFrccT6MrjL"
        if not self.api_host:
            raise ValueError("API_HOST must be provided as a parameter or environment variable.")

    def _get(self, path: str, params: dict = None) -> Any:
        url = f"{self.api_host}{path}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("data")


    def _post(self, path: str, json_body: dict) -> Any:
        url = f"{self.api_host}{path}"
        response = requests.post(url, json=json_body)
        response.raise_for_status()
        return response.json().get("data")

    def check_mpc_exists(self, src_addr: str) -> bool:
        return self._get(f"/mpc-wallet/{src_addr}")

    def get_mpc_wallet_id(self, src_addr: str) -> str:
        return self._get(f"/mpc-wallet-id/{src_addr}")

    def create_mpc_wallet(self, src_addr: str) -> str:
        return self._post("/mpc-wallet", {"srcAddr": src_addr})

    def find_mpc_addr(self, src_addr: str, chain: str) -> str:
        return self._get(f"/mpc-addr/{src_addr}/{chain}")

    def list_mpc_addr(self, src_addr: str, chain: str) -> Any:
        result = self._get(f"/mpc-addrs/{src_addr}/{chain}")
        return [MPCAddress(**item) for item in result] if result else []

    def estimate_tx_fee(self, wallet_id: str, token: str, amount: str, to_addr: str) -> float:
        return self._get("/estimate-tx-fee", {
            "walletId": wallet_id,
            "token": token,
            "amount": amount,
            "toAddr": to_addr
        })

    def create_maker_order(self, tx_hash: str, src_chain: str, target_chain: str,
                           i_token: str, i_amount: str, o_token: str, o_amount: str, slippage: str, sig: str) -> Any:
        return self._post("/maker-orders", {
            "txHash": tx_hash,
            "srcChain": src_chain,
            "targetChain": target_chain,
            "iToken": i_token,
            "iAmount": i_amount,
            "oToken": o_token,
            "oAmount": o_amount,
            "slippage": slippage,
            "sig": sig
        })

    def list_maker_orders(self) -> List[MakerOrder]:
        url = f"{self.api_host}/maker-orders"
        response = requests.get(url)
        response.raise_for_status()

        # 假设 response 是 JSON 数组
        data = response.json()

        return [MakerOrder(**item) for item in data]

    def create_taker_order(self, order_id: str, tx_hash: str) -> Any:
        return self._post("/taker-orders", {
            "orderId": order_id,
            "txHash": tx_hash
        })