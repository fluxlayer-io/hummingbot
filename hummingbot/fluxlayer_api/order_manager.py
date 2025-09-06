import asyncio
from typing import Dict, Any, Optional

from hummingbot.fluxlayer_api.binance_spot_direct import BinanceSpotDirectAPI
from hummingbot.fluxlayer_api.bybit_spot_direct import BybitSpotDirectAPI
from hummingbot.fluxlayer_api.okx_spot_direct import OKXSpotDirectAPI
from hummingbot.fluxlayer_api.hyperliquid_spot_direct import HyperliquidSpotDirectAPI


class OrderManager:

    def __init__(self):
        pass

    async def place_order(
        self,
        connector_name: str,
        trading_pair: str,
        amount: float,
        is_buy: bool,
    ) -> Dict[str, Any]:
        """
        参数:
            connector_name: 交易所名称 ("binance", "bybit", "okx", "hyperliquid")
            trading_pair: 交易对，如 "BTC-USDT" 或 "BTC-USDC"
            amount: 交易数量
            is_buy: True=买入, False=卖出
        返回:
            Dict[str, Any]: 下单结果
        """
        try:
            # 转换买卖方向
            side = "BUY" if is_buy else "SELL"
            
            print(f"🔧 [ORDER] {connector_name.upper()}: {side} {amount} {trading_pair}")
            
            # 根据交易所名称直接调用对应的直连 API
            if connector_name.lower() == "binance":
                api = BinanceSpotDirectAPI(debug=True)
                return api.place_market_order(side, trading_pair, amount)
                
            elif connector_name.lower() == "bybit":
                api = BybitSpotDirectAPI(debug=True)
                return api.place_market_order(side, trading_pair, amount)
                
            elif connector_name.lower() == "okx":
                api = OKXSpotDirectAPI(debug=True)
                return api.place_market_order(side, trading_pair, amount)
                
            elif connector_name.lower() == "hyperliquid":
                api = HyperliquidSpotDirectAPI(debug=True)
                return api.place_market_order(side, trading_pair, amount)
                
            else:
                error_msg = f"不支持的交易所: {connector_name}"
                print(f"❌ [ORDER ERROR] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "amount": amount,
                    "side": side
                }
                
        except Exception as e:
            error_msg = f"{connector_name} 下单失败: {str(e)}"
            print(f"❌ [ORDER ERROR] {error_msg}")
            
            # 打印详细的错误信息，方便调试
            import traceback
            print(f"❌ [ORDER ERROR] 详细错误信息:")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": side
            }

    async def get_order_details(self, connector_name: str, order_id: str, trading_pair: str) -> Dict[str, Any]:
        """
        查询订单详情 - 可选功能
        
        参数:
            connector_name: 交易所名称
            order_id: 订单ID
            trading_pair: 交易对
            
        返回:
            Dict[str, Any]: 订单详情
        """
        try:
            print(f"🔍 [ORDER DETAILS] {connector_name.upper()}: {order_id} {trading_pair}")
            
            # 根据交易所名称调用对应的 API
            if connector_name.lower() == "binance":
                api = BinanceSpotDirectAPI(debug=True)
                return api.get_order_details(order_id, trading_pair)
                
            elif connector_name.lower() == "bybit":
                api = BybitSpotDirectAPI(debug=True)
                return api.get_order_details(order_id, trading_pair)
                
            elif connector_name.lower() == "okx":
                api = OKXSpotDirectAPI(debug=True)
                return api.get_order_details(order_id, trading_pair)
                
            elif connector_name.lower() == "hyperliquid":
                api = HyperliquidSpotDirectAPI(debug=True)
                return api.get_order_details(order_id, trading_pair)
                
            else:
                error_msg = f"不支持的交易所: {connector_name}"
                print(f"❌ [ORDER DETAILS ERROR] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "exchange": connector_name
                }
                
        except Exception as e:
            error_msg = f"{connector_name} 查询订单详情失败: {str(e)}"
            print(f"❌ [ORDER DETAILS ERROR] {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "exchange": connector_name,
                "order_id": order_id,
                "trading_pair": trading_pair
            }

    async def get_account_balances(self, connector_name: str, **kwargs) -> Dict[str, Any]:
        """
        查询账户余额 - 可选功能
        
        参数:
            connector_name: 交易所名称
            **kwargs: 其他参数
            
        返回:
            Dict[str, Any]: 账户余额
        """
        try:
            print(f"💰 [BALANCES] {connector_name.upper()}")
            
            # 根据交易所名称调用对应的 API
            if connector_name.lower() == "binance":
                api = BinanceSpotDirectAPI(debug=True)
                return api.get_account_balances(**kwargs)
                
            elif connector_name.lower() == "bybit":
                api = BybitSpotDirectAPI(debug=True)
                return api.get_account_balances(**kwargs)
                
            elif connector_name.lower() == "okx":
                api = OKXSpotDirectAPI(debug=True)
                return api.get_account_balances(**kwargs)
                
            elif connector_name.lower() == "hyperliquid":
                api = HyperliquidSpotDirectAPI(debug=True)
                return api.get_account_balances(**kwargs)
                
            else:
                error_msg = f"不支持的交易所: {connector_name}"
                print(f"❌ [BALANCES ERROR] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "exchange": connector_name
                }
                
        except Exception as e:
            error_msg = f"{connector_name} 查询余额失败: {str(e)}"
            print(f"❌ [BALANCES ERROR] {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "exchange": connector_name
            }

    def get_supported_exchanges(self) -> list:
        """
        获取支持的交易所列表
        
        返回:
            list: 支持的交易所名称列表
        """
        return ["binance", "bybit", "okx", "hyperliquid"]


# 测试函数
async def test_order_manager():
    """测试订单管理器功能"""
    print("=== 测试极简版 OrderManager ===")
    
    order_manager = OrderManager()
    
    # 测试支持的交易所
    print(f"\n支持的交易所: {order_manager.get_supported_exchanges()}")
    
    # 测试下单（这里只是演示，不会真实下单）
    test_exchanges = [
        ("hyperliquid", "BTC-USDC", 0.0001),
        ("okx", "BTC-USDT", 0.0001), 
        ("bybit", "BTC-USDT", 0.0001),
        ("binance", "BTC-USDT", 0.0001)
    ]
    
    for exchange, pair, amount in test_exchanges:
        print(f"\n--- 测试 {exchange.upper()} ---")
        try:
            # 测试余额查询
            balance_result = await order_manager.get_account_balances(exchange)
            if balance_result.get("success"):
                print(f"✅ {exchange} 余额查询成功")
            else:
                print(f"❌ {exchange} 余额查询失败: {balance_result.get('error')}")
                
            # 这里不实际下单，只测试接口
            print(f"🔧 模拟 {exchange} {pair} 买单测试...")
            # result = await order_manager.place_order(exchange, pair, amount, True)
            
        except Exception as e:
            print(f"❌ 测试 {exchange} 时出错: {e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_order_manager())