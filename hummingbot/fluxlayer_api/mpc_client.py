import os
from dataclasses import dataclass
from typing import Optional, Any, List

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
    quota_id: Optional[int] = None

@dataclass
class MPCAddress:
    chain_id: str
    wallet_id: str
    address: str
    encoding: str

class MPCClient:
    def __init__(self, api_host: str = None):
        self.api_host = api_host or os.environ.get("MPC_API_HOST", "http://localhost:15887")
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

    def create_maker_order(
            self,
            src_chain: str,
            target_chain: str,
            i_token: str,
            i_amount: str,
            o_token: str,
            o_amount: str,
            wallet_id: str,
            tx_hash: Optional[str] = "fake",
            slippage: Optional[str] = "0.001",
            sig: Optional[str] = "fake",
            quota_id: Optional[int] = None
    ) -> Any:
        payload = {
            "srcChain": src_chain,
            "targetChain": target_chain,
            "iToken": i_token,
            "iAmount": i_amount,
            "oToken": o_token,
            "oAmount": o_amount,
            "walletId": wallet_id,
            "txHash": tx_hash,
            "slippage": slippage,
            "sig": sig
        }
        
        if quota_id is not None:
            payload["quotaId"] = quota_id

        return self._post("/maker-orders", payload)

    def list_maker_orders(self) -> List[MakerOrder]:
        url = f"{self.api_host}/maker-orders"
        response = requests.get(url)
        response.raise_for_status()

        # 假设 response 是 JSON 数组
        data = response.json()

        return [MakerOrder(**item) for item in data]

    def create_taker_order(self, order_id: str, wallet_id: str, tx_hash: Optional[str] = "fake") -> Any:
        return self._post("/taker-orders", {
            "orderId": order_id,
            "txHash": tx_hash,
            "walletId": wallet_id
        })

    def create_quota(self, solver_id: int, source_amount: str, source_price: str, target_amount: str, target_price: str) -> Any:
        """创建报价记录"""
        payload = {
            "solverId": solver_id,
            "sourceAmount": source_amount,
            "sourcePrice": source_price,
            "targetAmount": target_amount,
            "targetPrice": target_price
        }
        return self._post("/quota", payload)

    def get_quota(self, quota_id: int) -> Any:
        """获取报价信息"""
        return self._get(f"/quota/{quota_id}")

    def create_cex_quota_mapping(self, cex_connector_id: str, quota_id: int) -> Any:
        """创建 CEX-Quota 映射"""
        payload = {
            "cexConnectorId": cex_connector_id,
            "quotaId": quota_id
        }
        return self._post("/cex-quota-mapping", payload)

    def get_quota_by_cex_connector(self, cex_connector_id: str) -> Any:
        """通过 CEX connector 获取报价"""
        return self._get(f"/cex-quota/{cex_connector_id}")
    
    def get_cex_connector_by_quota(self, quota_id: int) -> Optional[str]:
        """通过 quota_id 获取对应的 CEX connector ID"""
        try:
            result = self._get(f"/quota/{quota_id}/cex")
            return result if isinstance(result, str) else None
        except Exception:
            return None