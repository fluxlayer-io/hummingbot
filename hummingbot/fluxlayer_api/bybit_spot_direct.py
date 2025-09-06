"""
Bybit Spot 直连交易 API 实现
使用 pybit 库简化实现，避免复杂的签名算法

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
    from pybit.unified_trading import HTTP
    from pybit.exceptions import InvalidRequestError, FailedRequestError
except ImportError:
    print("❌ [BYBIT ERROR] pybit library not installed. Please install it with: pip install pybit")
    raise


class BybitSpotDirectAPI:
    """
    Bybit Spot 现货交易直连 API
    使用 pybit 库实现，避免手动签名算法
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, debug: bool = False):
        """
        初始化 Bybit Spot API 客户端
        
        参数:
            api_key: Bybit API Key，如果为空则从环境变量读取
            api_secret: Bybit API Secret，如果为空则从环境变量读取
            debug: 是否开启调试模式
        """
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        self.debug = debug
        self.base_url = "https://api.bybit.com"
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required. Set them via parameters or environment variables.")
        
        # 初始化 pybit 客户端
        try:
            self.client = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=False
            )
            
            if self.debug:
                print(f"🔧 [BYBIT INIT] Client initialized successfully")
                print(f"🔧 [BYBIT INIT] API Key: {self.api_key[:8]}...")
                print(f"🔧 [BYBIT INIT] Base URL: {self.base_url}")
                
        except Exception as e:
            if self.debug:
                print(f"❌ [BYBIT INIT ERROR] Failed to initialize client: {str(e)}")
            raise
    
    def _format_trading_pair(self, pair: str) -> str:
        """
        格式化交易对，从 BTC-USDT 转换为 BTCUSDT
        
        参数:
            pair: 交易对，如 "BTC-USDT"
            
        返回:
            str: Bybit 格式的交易对，如 "BTCUSDT"
        """
        formatted = pair.replace("-", "").upper()
        
        if self.debug:
            print(f"🔄 [BYBIT FORMAT] {pair} -> {formatted}")
            
        return formatted
    
    def _format_quantity(self, amount: float) -> str:
        """
        格式化数量，确保精度符合 Bybit 要求
        
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
            print(f"🔢 [BYBIT QUANTITY] {amount} -> {formatted}")
            
        return formatted
    
    def get_ticker_price(self, pair: str) -> Dict[str, Any]:
        """
        获取交易对的实时价格
        
        参数:
            pair: 交易对，如 "BTC-USDT"
            
        返回:
            Dict[str, Any]: 价格信息
        """
        try:
            symbol = self._format_trading_pair(pair)
            
            if self.debug:
                print(f"🔧 [BYBIT GET PRICE] Getting price for pair: {pair} -> {symbol}")
            
            # 使用 pybit 库获取ticker价格
            ticker_result = self.client.get_tickers(
                category="spot",
                symbol=symbol
            )
            
            if self.debug:
                print(f"🔍 [BYBIT GET PRICE RESPONSE] {json.dumps(ticker_result, indent=2)}")
            
            # 检查响应状态
            if ticker_result.get("retCode") == 0:
                result = ticker_result.get("result", {})
                ticker_list = result.get("list", [])
                
                if ticker_list:
                    ticker_data = ticker_list[0]
                    last_price = float(ticker_data.get("lastPrice", "0"))
                    
                    if self.debug:
                        print(f"✅ [BYBIT GET PRICE SUCCESS] Current price: {last_price}")
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "price": last_price,
                        "exchange": "bybit",
                        "trading_pair": pair,
                        "raw_response": ticker_result
                    }
                else:
                    return {
                        "success": False,
                        "error": "Price data not found",
                        "exchange": "bybit",
                        "trading_pair": pair
                    }
            else:
                error_msg = f"Bybit API error (code: {ticker_result.get('retCode')}): {ticker_result.get('retMsg')}"
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": ticker_result.get('retCode'),
                    "exchange": "bybit",
                    "trading_pair": pair
                }
                
        except Exception as e:
            error_msg = f"Error getting ticker price: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT GET PRICE ERROR] {error_msg}")
                import traceback
                print(f"❌ [BYBIT GET PRICE ERROR] Traceback: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit",
                "trading_pair": pair
            }
    
    def place_market_order(self, side: str, pair: str, amount: float) -> Dict[str, Any]:
        """
        下市价单
        
        参数:
            side: 买卖方向，"BUY" 或 "SELL"
            pair: 交易对，如 "BTC-USDT"
            amount: BTC 数量（始终为基础货币数量）
            
        返回:
            Dict[str, Any]: 下单结果
        """
        try:
            symbol = self._format_trading_pair(pair)
            side = side.upper()
            
            # 转换为 Bybit API 格式
            bybit_side = "Buy" if side == "BUY" else "Sell"
            
            # Bybit 买单需要 USDT 数量，卖单需要 BTC 数量
            if bybit_side == "Buy":
                # 买单：需要将 BTC 数量转换为 USDT 数量
                if self.debug:
                    print(f"🔧 [BYBIT BUY ORDER] Getting current price for conversion...")
                
                price_result = self.get_ticker_price(pair)
                if not price_result.get("success"):
                    return {
                        "success": False,
                        "error": f"Failed to get price for conversion: {price_result.get('error')}",
                        "exchange": "bybit",
                        "trading_pair": pair,
                        "amount": amount,
                        "side": side
                    }
                
                current_price = price_result["price"]
                usdt_amount = amount * current_price  # BTC数量 × BTC价格 = USDT数量
                quantity = self._format_quantity(usdt_amount)
                
                if self.debug:
                    print(f"🔧 [BYBIT BUY ORDER] BTC Amount: {amount}, Price: {current_price}, USDT Amount: {usdt_amount}")
                    print(f"🔧 [BYBIT BUY ORDER] Converted BTC quantity to USDT quantity: {quantity}")
            else:
                # 卖单：直接使用 BTC 数量
                quantity = self._format_quantity(amount)
                
                if self.debug:
                    print(f"🔧 [BYBIT SELL ORDER] Using BTC quantity directly: {quantity}")
            
            if self.debug:
                print(f"🔧 [BYBIT MARKET ORDER] Side: {side} -> {bybit_side}, Pair: {pair} -> {symbol}, Amount: {amount}")
                print(f"🔍 [BYBIT ORDER PARAMS] Symbol: {symbol}, Side: {bybit_side}, Quantity: {quantity}")
            
            # 使用 pybit 库的市价单方法
            order_result = self.client.place_order(
                category="spot",
                symbol=symbol,
                side=bybit_side,
                orderType="Market",
                qty=quantity
            )
            
            if self.debug:
                print(f"✅ [BYBIT ORDER SUCCESS] Order placed successfully")
                print(f"🔍 [BYBIT ORDER RESPONSE] {json.dumps(order_result, indent=2)}")
            
            # 检查响应状态
            if order_result.get("retCode") == 0:
                result = order_result.get("result", {})
                
                # 统一返回格式
                return {
                    "success": True,
                    "order_id": result.get("orderId"),
                    "order_link_id": result.get("orderLinkId"),
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": quantity,
                    "status": "SUBMITTED",
                    "exchange": "bybit",
                    "trading_pair": pair,
                    "amount": amount,
                    "raw_response": order_result
                }
            else:
                error_msg = f"Bybit API error (code: {order_result.get('retCode')}): {order_result.get('retMsg')}"
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": order_result.get('retCode'),
                    "exchange": "bybit",
                    "trading_pair": pair,
                    "amount": amount,
                    "side": side
                }
            
        except InvalidRequestError as e:
            error_msg = f"Bybit invalid request error: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT ORDER ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }
            
        except FailedRequestError as e:
            error_msg = f"Bybit failed request error: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT ORDER ERROR] {error_msg}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }
            
        except Exception as e:
            error_msg = f"Error placing market order: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [BYBIT ORDER ERROR] Traceback: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit",
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
                print(f"🔧 [BYBIT GET ORDER] Order ID: {order_id}, Pair: {pair} -> {symbol}")
            
            # 使用 pybit 库获取订单详情
            order_result = self.client.get_open_orders(
                category="spot",
                symbol=symbol,
                orderId=order_id
            )
            
            if self.debug:
                print(f"🔍 [BYBIT GET ORDER RESPONSE] {json.dumps(order_result, indent=2)}")
            
            # 检查响应状态
            if order_result.get("retCode") == 0:
                result = order_result.get("result", {})
                order_list = result.get("list", [])
                
                if order_list:
                    order_details = order_list[0]
                    
                    if self.debug:
                        print(f"✅ [BYBIT GET ORDER SUCCESS] Order details retrieved")
                        print(f"🔍 [BYBIT ORDER DETAILS] {json.dumps(order_details, indent=2)}")
                    
                    # 统一返回格式
                    return {
                        "success": True,
                        "order_id": order_details.get("orderId"),
                        "order_link_id": order_details.get("orderLinkId"),
                        "symbol": order_details.get("symbol"),
                        "side": order_details.get("side"),
                        "type": order_details.get("orderType"),
                        "quantity": order_details.get("qty"),
                        "executed_quantity": order_details.get("cumExecQty"),
                        "price": order_details.get("price"),
                        "average_price": order_details.get("avgPrice"),
                        "status": order_details.get("orderStatus"),
                        "create_time": order_details.get("createdTime"),
                        "update_time": order_details.get("updatedTime"),
                        "exchange": "bybit",
                        "trading_pair": pair,
                        "raw_response": order_result
                    }
                else:
                    return {
                        "success": False,
                        "error": "Order not found",
                        "exchange": "bybit",
                        "trading_pair": pair,
                        "order_id": order_id
                    }
            else:
                error_msg = f"Bybit API error (code: {order_result.get('retCode')}): {order_result.get('retMsg')}"
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": order_result.get('retCode'),
                    "exchange": "bybit",
                    "trading_pair": pair,
                    "order_id": order_id
                }
                
        except Exception as e:
            error_msg = f"Error getting order details: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT GET ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [BYBIT GET ORDER ERROR] Traceback: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit",
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
                print(f"🔧 [BYBIT GET BALANCES] Fetching account balances...")
                print(f"🔧 [BYBIT GET BALANCES] Show zero balances: {show_zero_balances}")
                print(f"🔧 [BYBIT GET BALANCES] Min balance threshold: {min_balance}")
            
            # 使用 pybit 库获取账户余额
            balance_result = self.client.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if self.debug:
                print(f"🔍 [BYBIT GET BALANCES RESPONSE] {json.dumps(balance_result, indent=2)}")
            
            # 检查响应状态
            if balance_result.get("retCode") == 0:
                result = balance_result.get("result", {})
                account_list = result.get("list", [])
                
                if self.debug:
                    print(f"✅ [BYBIT GET BALANCES SUCCESS] Account info retrieved")
                    print(f"🔍 [BYBIT ACCOUNT COUNT] Found {len(account_list)} account(s)")
                
                balances = {}
                total_assets = 0
                assets_with_balance = 0
                assets_with_significant_balance = 0
                
                # 处理账户余额数据
                for account in account_list:
                    coin_list = account.get("coin", [])
                    account_type = account.get("accountType", "UNKNOWN")
                    
                    if self.debug:
                        print(f"🔍 [BYBIT ACCOUNT] Type: {account_type}, Coins: {len(coin_list)}")
                    
                    for coin_data in coin_list:
                        asset = coin_data.get("coin", "")
                        wallet_balance = float(coin_data.get("walletBalance", "0") or "0")
                        equity = float(coin_data.get("equity", "0") or "0")
                        available_balance = float(coin_data.get("availableToWithdraw", "0") or "0")
                        
                        total = wallet_balance
                        total_assets += 1
                        
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
                                "wallet_balance": str(wallet_balance),
                                "equity": str(equity),
                                "available_balance": str(available_balance),
                                "total": str(total),
                                "wallet_balance_float": wallet_balance,
                                "equity_float": equity,
                                "available_balance_float": available_balance,
                                "total_float": total,
                                "account_type": account_type
                            }
                
                # 构建返回结果
                result = {
                    "success": True,
                    "exchange": "bybit",
                    "account_type": "UNIFIED",
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
                    print(f"✅ [BYBIT GET BALANCES SUCCESS] Total assets: {total_assets}")
                    print(f"✅ [BYBIT GET BALANCES SUCCESS] Assets with any balance: {assets_with_balance}")
                    print(f"✅ [BYBIT GET BALANCES SUCCESS] Assets with significant balance (>={min_balance}): {assets_with_significant_balance}")
                    print(f"✅ [BYBIT GET BALANCES SUCCESS] Displayed assets: {len(balances)}")
                    
                    # 显示前几个显示的资产
                    displayed_assets = list(balances.keys())[:5]
                    if displayed_assets:
                        print(f"✅ [BYBIT GET BALANCES SUCCESS] Top displayed assets: {displayed_assets}")
                
                return result
            else:
                error_msg = f"Bybit API error (code: {balance_result.get('retCode')}): {balance_result.get('retMsg')}"
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": balance_result.get('retCode'),
                    "exchange": "bybit"
                }
                
        except Exception as e:
            error_msg = f"Error getting account balances: {str(e)}"
            if self.debug:
                print(f"❌ [BYBIT GET BALANCES ERROR] {error_msg}")
                import traceback
                print(f"❌ [BYBIT GET BALANCES ERROR] Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": error_msg,
                "exchange": "bybit"
            }


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description='Bybit Spot Trading API',
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
        api = BybitSpotDirectAPI(debug=args.debug)
        
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