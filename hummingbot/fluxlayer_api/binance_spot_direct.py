"""
Binance Spot 直连交易 API 实现
使用 python-binance 库简化实现，避免复杂的签名算法

支持功能:
1. 市价单下单 (place_market_order)
2. 订单详情查询 (get_order_details)  
3. 账户余额查询 (get_account_balances)
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Optional
from decimal import Decimal

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except ImportError:
    print("❌ [BINANCE ERROR] python-binance library not installed. Please install it with: pip install python-binance")
    raise


class BinanceSpotDirectAPI:
    """
    Binance Spot 现货交易直连 API
    使用 python-binance 库实现，避免手动签名算法
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, debug: bool = False):
        """
        初始化 Binance Spot API 客户端
        
        参数:
            api_key: Binance API Key，如果为空则从环境变量读取
            api_secret: Binance API Secret，如果为空则从环境变量读取  
            debug: 是否开启调试模式
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.debug = debug
        self.base_url = "https://api.binance.com"
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required. Set them via parameters or environment variables.")
        
        # 初始化 python-binance 客户端
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=False  # 使用正式环境
            )
            
            if self.debug:
                print(f"🔧 [BINANCE INIT] Client initialized successfully")
                print(f"🔧 [BINANCE INIT] API Key: {self.api_key[:8]}...")
                print(f"🔧 [BINANCE INIT] Base URL: {self.base_url}")
                
        except Exception as e:
            if self.debug:
                print(f"❌ [BINANCE INIT ERROR] Failed to initialize client: {str(e)}")
            raise
    
    def _format_trading_pair(self, pair: str) -> str:
        """
        格式化交易对，从 BTC-USDT 转换为 BTCUSDT
        
        参数:
            pair: 交易对，如 "BTC-USDT"
            
        返回:
            str: Binance 格式的交易对，如 "BTCUSDT"
        """
        formatted = pair.replace("-", "").upper()
        
        if self.debug:
            print(f"🔄 [BINANCE FORMAT] {pair} -> {formatted}")
            
        return formatted
    
    def _format_quantity(self, amount: float) -> str:
        """
        格式化数量，确保精度符合 Binance 要求
        
        参数:
            amount: 数量
            
        返回:
            str: 格式化后的数量字符串
        """
        # 使用 Decimal 确保精度
        decimal_amount = Decimal(str(amount))
        
        # 移除尾随零
        formatted = f"{decimal_amount:f}".rstrip('0').rstrip('.')
        
        if self.debug:
            print(f"🔢 [BINANCE QUANTITY] {amount} -> {formatted}")
            
        return formatted
    
    def place_market_order(self, side: str, pair: str, amount: float) -> Dict[str, Any]:
        """
        下市价单
        
        参数:
            side: 买卖方向，"BUY" 或 "SELL"
            pair: 交易对，如 "BTC-USDT"
            amount: 交易数量
            
        返回:
            Dict[str, Any]: 下单结果
        """
        try:
            symbol = self._format_trading_pair(pair)
            quantity = self._format_quantity(amount)
            side = side.upper()
            
            if self.debug:
                print(f"🔧 [BINANCE MARKET ORDER] Side: {side}, Pair: {pair} -> {symbol}, Amount: {amount}")
                print(f"🔍 [BINANCE ORDER PARAMS] Symbol: {symbol}, Side: {side}, Quantity: {quantity}")
            
            # 使用 python-binance 库的市价单方法
            if side == "BUY":
                order_result = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            elif side == "SELL":
                order_result = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            else:
                raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
            
            if self.debug:
                print(f"✅ [BINANCE ORDER SUCCESS] Order placed successfully")
                print(f"🔍 [BINANCE ORDER RESPONSE] {json.dumps(order_result, indent=2)}")
            
            # 统一返回格式
            return {
                "success": True,
                "order_id": str(order_result.get("orderId")),
                "client_order_id": order_result.get("clientOrderId"),
                "symbol": order_result.get("symbol"),
                "side": order_result.get("side"),
                "type": order_result.get("type"),
                "quantity": order_result.get("origQty"),
                "executed_quantity": order_result.get("executedQty"),
                "price": order_result.get("price"),
                "status": order_result.get("status"),
                "transaction_time": order_result.get("transactTime"),
                "exchange": "binance",
                "trading_pair": pair,
                "amount": amount,
                "raw_response": order_result
            }
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error (code: {e.code}): {e.message}"
            if self.debug:
                print(f"❌ [BINANCE ORDER ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "error_code": e.code,
                "exchange": "binance",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }
            
        except BinanceOrderException as e:
            error_msg = f"Binance order error (code: {e.code}): {e.message}"
            if self.debug:
                print(f"❌ [BINANCE ORDER ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "error_code": e.code,
                "exchange": "binance",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }
            
        except Exception as e:
            error_msg = f"Error placing market order: {str(e)}"
            if self.debug:
                print(f"❌ [BINANCE ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [BINANCE ORDER ERROR] Traceback: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "binance",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }
    
    def get_order_details(self, order_id: str, pair: str) -> Dict[str, Any]:
        """
        获取订单详情
        
        参数:
            order_id: 订单ID
            pair: 交易对，如 "BTC-USDT"
            
        返回:
            Dict[str, Any]: 订单详情
        """
        try:
            symbol = self._format_trading_pair(pair)
            
            if self.debug:
                print(f"🔧 [BINANCE GET ORDER] Order ID: {order_id}, Pair: {pair} -> {symbol}")
            
            # 使用 python-binance 库获取订单详情
            order_details = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            
            if self.debug:
                print(f"✅ [BINANCE GET ORDER SUCCESS] Order details retrieved")
                print(f"🔍 [BINANCE ORDER DETAILS] {json.dumps(order_details, indent=2)}")
            
            # 统一返回格式
            return {
                "success": True,
                "order_id": str(order_details.get("orderId")),
                "client_order_id": order_details.get("clientOrderId"),
                "symbol": order_details.get("symbol"),
                "side": order_details.get("side"),
                "type": order_details.get("type"),
                "quantity": order_details.get("origQty"),
                "executed_quantity": order_details.get("executedQty"),
                "price": order_details.get("price"),
                "average_price": order_details.get("avgPrice"),
                "status": order_details.get("status"),
                "time": order_details.get("time"),
                "update_time": order_details.get("updateTime"),
                "exchange": "binance",
                "trading_pair": pair,
                "raw_response": order_details
            }
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error (code: {e.code}): {e.message}"
            if self.debug:
                print(f"❌ [BINANCE GET ORDER ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "error_code": e.code,
                "exchange": "binance",
                "trading_pair": pair,
                "order_id": order_id
            }
            
        except Exception as e:
            error_msg = f"Error getting order details: {str(e)}"
            if self.debug:
                print(f"❌ [BINANCE GET ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [BINANCE GET ORDER ERROR] Traceback: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "binance",
                "trading_pair": pair,
                "order_id": order_id
            }
    
    def get_account_balances(self, show_zero_balances: bool = False, min_balance: float = 0.0) -> Dict[str, Any]:
        """
        获取账户余额
        
        参数:
            show_zero_balances: 是否显示零余额的资产
            min_balance: 最小余额阈值，小于此值的余额将被过滤（当show_zero_balances=False时）
            
        返回:
            Dict[str, Any]: 账户余额信息
        """
        try:
            if self.debug:
                print(f"🔧 [BINANCE GET BALANCES] Fetching account balances...")
                print(f"🔧 [BINANCE GET BALANCES] Show zero balances: {show_zero_balances}")
                print(f"🔧 [BINANCE GET BALANCES] Min balance threshold: {min_balance}")
            
            # 使用 python-binance 库获取账户信息
            account_info = self.client.get_account()
            
            if self.debug:
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Account info retrieved")
                print(f"🔍 [BINANCE ACCOUNT INFO] Can trade: {account_info.get('canTrade')}")
                print(f"🔍 [BINANCE ACCOUNT INFO] Can withdraw: {account_info.get('canWithdraw')}")
                print(f"🔍 [BINANCE ACCOUNT INFO] Can deposit: {account_info.get('canDeposit')}")
            
            # 处理余额数据
            raw_balances = account_info.get("balances", [])
            balances = {}
            
            total_assets = len(raw_balances)
            assets_with_balance = 0
            assets_with_significant_balance = 0
            
            for balance in raw_balances:
                asset = balance.get("asset")
                free = float(balance.get("free", "0"))
                locked = float(balance.get("locked", "0"))
                total = free + locked
                
                if total > 0:
                    assets_with_balance += 1
                
                if total >= min_balance:
                    assets_with_significant_balance += 1
                
                # 根据过滤条件决定是否包含此资产
                should_include = True
                
                if not show_zero_balances:
                    # 不显示零余额，需要检查最小余额阈值
                    if total <= min_balance:
                        should_include = False
                
                if should_include:
                    balances[asset] = {
                        "free": str(free),
                        "locked": str(locked),
                        "total": str(total),
                        "free_float": free,
                        "locked_float": locked, 
                        "total_float": total
                    }
            
            # 构建返回结果
            result = {
                "success": True,
                "exchange": "binance",
                "account_type": account_info.get("accountType", "SPOT"),
                "can_trade": account_info.get("canTrade", False),
                "can_withdraw": account_info.get("canWithdraw", False),
                "can_deposit": account_info.get("canDeposit", False),
                "balances": balances,
                "summary": {
                    "total_assets": total_assets,
                    "assets_with_balance": assets_with_balance,
                    "assets_with_significant_balance": assets_with_significant_balance,
                    "displayed_assets": len(balances),
                    "show_zero_balances": show_zero_balances,
                    "min_balance_threshold": min_balance
                }
            }
            
            if self.debug:
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Total assets: {total_assets}")
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Assets with any balance: {assets_with_balance}")
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Assets with significant balance (>={min_balance}): {assets_with_significant_balance}")
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Displayed assets: {len(balances)}")
                print(f"✅ [BINANCE GET BALANCES SUCCESS] Account type: {account_info.get('accountType')}")
                
                # 显示前几个显示的资产
                displayed_assets = list(balances.keys())[:5]
                if displayed_assets:
                    print(f"✅ [BINANCE GET BALANCES SUCCESS] Top displayed assets: {displayed_assets}")
            
            return result
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error (code: {e.code}): {e.message}"
            if self.debug:
                print(f"❌ [BINANCE GET BALANCES ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "error_code": e.code,
                "exchange": "binance"
            }
            
        except Exception as e:
            error_msg = f"Error getting account balances: {str(e)}"
            if self.debug:
                print(f"❌ [BINANCE GET BALANCES ERROR] {error_msg}")
                import traceback
                print(f"❌ [BINANCE GET BALANCES ERROR] Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": error_msg,
                "exchange": "binance"
            }


def test_binance_spot_api():
    """测试 Binance Spot API 功能（同步版本）"""
    print("=== 测试 Binance Spot API ===")
    
    try:
        # 创建 API 实例
        api = BinanceSpotDirectAPI(debug=True)

        print("\n1. 测试账户余额查询...")
        balance_result = api.get_account_balances(show_zero_balances=False, min_balance=0.00001)
        print(f"余额查询结果: {json.dumps(balance_result, indent=2, ensure_ascii=False)}")

        print("\n2. 测试市价买单...")
        buy_result = api.place_market_order("BUY", "BTC-USDT", 0.0001)
        print(f"买单结果: {json.dumps(buy_result, indent=2, ensure_ascii=False)}")

        if buy_result.get("success"):
            order_id = buy_result["order_id"]
            print(f"\n3. 测试获取订单详情 (Order ID: {order_id})...")

            # 等待一小段时间让订单状态更新
            time.sleep(2)

            order_details = api.get_order_details(order_id, "BTC-USDT")
            print(f"订单详情: {json.dumps(order_details, indent=2, ensure_ascii=False)}")

        # print("\n4. 测试市价卖单...")
        # sell_result = api.place_market_order("SELL", "BTC-USDT", 0.0001)
        # print(f"卖单结果: {json.dumps(sell_result, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()




def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description='Binance Spot Trading API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 查询账户余额
  python %(prog)s --action balance
  
  # 下买单
  python %(prog)s --action order --pair BTC-USDT --amount 0.001 --side buy
  
  # 下卖单
  python %(prog)s --action order --pair BTC-USDT --amount 0.001 --side sell
        """)
    
    parser.add_argument('--action', '-a', 
                       choices=['balance', 'order'], 
                       required=True,
                       help='操作类型: balance(查询余额) 或 order(下单)')
    
    parser.add_argument('--pair', '-p', 
                       help='交易对 (例如: BTC-USDT)')
    
    parser.add_argument('--amount', '-amt', 
                       type=float,
                       help='交易数量')
    
    parser.add_argument('--side', '-s',
                       choices=['buy', 'sell'],
                       help='买卖方向: buy 或 sell')
    
    parser.add_argument('--debug', '-d',
                       action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        # 创建 API 实例
        api = BinanceSpotDirectAPI(debug=args.debug)
        
        if args.action == 'balance':
            print("=== 查询账户余额 ===")
            result = api.get_account_balances()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        elif args.action == 'order':
            # 验证下单所需参数
            if not args.pair:
                print("❌ 错误: 下单需要指定交易对 (--pair)")
                return 1
            if not args.amount:
                print("❌ 错误: 下单需要指定数量 (--amount)")
                return 1
            if not args.side:
                print("❌ 错误: 下单需要指定买卖方向 (--side)")
                return 1
                
            print(f"=== 执行{args.side.upper()}单 ===")
            print(f"交易对: {args.pair}")
            print(f"数量: {args.amount}")
            print(f"方向: {args.side.upper()}")
            
            result = api.place_market_order(args.side.upper(), args.pair, args.amount)
            print("下单结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 如果下单成功，查询订单详情
            if result.get("success"):
                order_id = result["order_id"]
                print(f"\n=== 订单详情 (Order ID: {order_id}) ===")
                time.sleep(1)  # 等待订单状态更新
                order_details = api.get_order_details(order_id, args.pair)
                print(json.dumps(order_details, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        import traceback
        if args.debug:
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())