#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional
from decimal import Decimal

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account import Account
except ImportError:
    print("❌ [HYPERLIQUID ERROR] hyperliquid-python-sdk library not installed. Please install it with: pip install hyperliquid-python-sdk")
    raise


class HyperliquidSpotDirectAPI:
    """
    Hyperliquid 现货直连交易 API
    
    特点：
    1. 统一接口设计，与 Binance/Bybit/OKX 保持一致
    2. BTC 特殊映射：BTC-USDC -> UBTC-USDC (内部处理，对外透明)
    3. 支持余额查询、价格获取、市价单交易、订单详情查询
    """

    def __init__(self, api_key: str = None, api_secret: str = None, debug: bool = False):
        """
        初始化 Hyperliquid API 客户端
        
        Args:
            api_key: Hyperliquid API Key (钱包地址)，如果为空则从环境变量读取
            api_secret: Hyperliquid API Secret (私钥)，如果为空则从环境变量读取
            debug: 是否开启调试模式
        """
        self.debug = debug
        
        try:
            # 从环境变量或参数获取认证信息
            self.api_key = api_key or os.getenv("HYPERLIQUID_API_KEY")
            self.api_secret = api_secret or os.getenv("HYPERLIQUID_API_SECRET")
            
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret are required. Set HYPERLIQUID_API_KEY and HYPERLIQUID_API_SECRET environment variables.")
            
            # 从私钥创建以太坊账户（Hyperliquid 使用以太坊兼容签名）
            self.account = Account.from_key(self.api_secret)
            self.user_address = self.account.address
            
            # 验证 API Key 是否匹配派生的地址
            if self.api_key.lower() != self.user_address.lower():
                raise ValueError(f"API key {self.api_key} doesn't match address derived from secret {self.user_address}")
            
            
            # 使用 mainnet API
            self.base_url = constants.MAINNET_API_URL
            
            # 初始化 Info 和 Exchange 实例
            self.info = Info(self.base_url, skip_ws=True)
            self.exchange = Exchange(self.account, self.base_url, meta=None)
            
            if self.debug:
                print(f"🔧 [HYPERLIQUID INIT] Client initialized successfully")
                print(f"🔧 [HYPERLIQUID INIT] API Key: {self.api_key}")
                print(f"🔧 [HYPERLIQUID INIT] User Address: {self.user_address}")
                print(f"🔧 [HYPERLIQUID INIT] Base URL: {self.base_url}")
            
        except Exception as e:
            if self.debug:
                print(f"❌ [HYPERLIQUID INIT ERROR] Failed to initialize client: {e}")
            raise

    def _format_trading_pair(self, pair: str) -> str:
        """
        格式化交易对，专门处理 BTC 映射到 Hyperliquid 现货格式
        
        BTC-USDC -> UBTC/USDC (UBTC 现货交易对，asset ID 10142)
        
        Args:
            pair: 原始交易对，如 "BTC-USDC"
            
        Returns:
            格式化后的 Hyperliquid 现货交易对，如 "UBTC/USDC"
        """
        formatted = pair.upper()
        
        # 处理 BTC 特殊映射，转换为 Hyperliquid 现货格式
        if formatted == "BTC-USDC":
            mapped_pair = "UBTC/USDC"  # UBTC 现货交易对，对应 asset ID 10142
            if self.debug:
                print(f"🔄 [HYPERLIQUID BTC MAPPING] {formatted} -> {mapped_pair} (UBTC 现货)")
            return mapped_pair
        
        # 其他交易对需要根据实际情况映射到对应的 @数字 格式
        # 目前只支持 BTC-USDC
        if self.debug:
            print(f"⚠️ [HYPERLIQUID PAIR WARNING] Unsupported pair: {formatted}")
        
        return formatted

    def _get_trading_pair_name(self, pair: str) -> str:
        """
        获取 Hyperliquid 完整交易对名称
        
        BTC-USDC -> UBTC/USDC
        
        Args:
            pair: 交易对，如 "BTC-USDC"
            
        Returns:
            Hyperliquid 交易对名称，如 "UBTC/USDC"
        """
        hyperliquid_pair = self._format_trading_pair(pair)
        
        if self.debug:
            print(f"🪙 [HYPERLIQUID TRADING PAIR] {pair} -> {hyperliquid_pair}")
        
        return hyperliquid_pair

    def _format_quantity(self, amount: float) -> str:
        """
        格式化数量，移除科学计数法
        
        Args:
            amount: 原始数量
            
        Returns:
            格式化后的数量字符串
        """
        if amount == 0:
            return "0"
        
        # 使用 Decimal 避免浮点数精度问题
        decimal_amount = Decimal(str(amount))
        
        # 格式化为字符串，移除尾随零
        formatted = format(decimal_amount, 'f')
        
        if self.debug:
            print(f"🔢 [HYPERLIQUID QUANTITY] {amount} -> {formatted}")
        
        return formatted

    def get_account_balances(self, show_zero_balances: bool = False, min_balance: float = 0.0) -> Dict[str, Any]:
        """
        获取账户余额
        
        Args:
            show_zero_balances: 是否显示零余额资产
            min_balance: 最小余额阈值
            
        Returns:
            包含余额信息的字典
        """
        try:
            if self.debug:
                print(f"🔧 [HYPERLIQUID GET BALANCES] Fetching account balances...")
                print(f"🔧 [HYPERLIQUID GET BALANCES] User Address: {self.user_address}")
                print(f"🔧 [HYPERLIQUID GET BALANCES] Show zero balances: {show_zero_balances}")
                print(f"🔧 [HYPERLIQUID GET BALANCES] Min balance threshold: {min_balance}")

            # 获取现货用户状态
            user_state = self.info.spot_user_state(self.user_address)
            
            if self.debug:
                print(f"🔍 [HYPERLIQUID GET BALANCES RESPONSE] {json.dumps(user_state, indent=2)}")

            balances = {}
            total_assets = 0
            assets_with_balance = 0
            
            # 解析现货余额信息
            if 'balances' in user_state:
                balance_list = user_state['balances']
                total_assets = len(balance_list)
                
                if self.debug:
                    print(f"🔍 [HYPERLIQUID ACCOUNT] Found {total_assets} balance(s)")
                
                for balance_item in balance_list:
                    coin = balance_item.get('coin', '')
                    hold_str = balance_item.get('hold', '0')
                    total_str = balance_item.get('total', '0')
                    
                    try:
                        hold = float(hold_str)
                        total = float(total_str)
                    except (ValueError, TypeError):
                        hold = 0.0
                        total = 0.0
                    
                    # 应用过滤逻辑
                    if not show_zero_balances and total <= min_balance:
                        continue
                    
                    if total > 0:
                        assets_with_balance += 1
                    
                    balances[coin] = {
                        "total": str(total),
                        "total_float": total,
                        "available_balance": str(hold),  # hold 是可用余额
                        "frozen_balance": str(total - hold),  # 冻结 = 总量 - 可用
                        "available_balance_float": hold,
                        "frozen_balance_float": total - hold
                    }

            if self.debug:
                print(f"✅ [HYPERLIQUID GET BALANCES SUCCESS] Account info retrieved")
                print(f"✅ [HYPERLIQUID GET BALANCES SUCCESS] Total assets: {total_assets}")
                print(f"✅ [HYPERLIQUID GET BALANCES SUCCESS] Assets with balance: {assets_with_balance}")
                print(f"✅ [HYPERLIQUID GET BALANCES SUCCESS] Displayed assets: {len(balances)}")

            return {
                "success": True,
                "exchange": "hyperliquid",
                "account_type": "SPOT",
                "user_address": self.user_address,
                "balances": balances,
                "summary": {
                    "total_assets": total_assets,
                    "assets_with_balance": assets_with_balance,
                    "displayed_assets": len(balances),
                    "show_zero_balances": show_zero_balances,
                    "min_balance_threshold": min_balance
                }
            }

        except Exception as e:
            error_msg = f"Error getting account balances: {e}"
            if self.debug:
                print(f"❌ [HYPERLIQUID GET BALANCES ERROR] {error_msg}")
                import traceback
                print(f"❌ [HYPERLIQUID GET BALANCES ERROR] Traceback: {traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "exchange": "hyperliquid",
                "account_type": "SPOT",
                "user_address": self.user_address,
                "balances": {}
            }

    def get_ticker_price(self, pair: str) -> Dict[str, Any]:
        """
        获取交易对价格
        
        Args:
            pair: 交易对，如 "BTC-USDC"
            
        Returns:
            包含价格信息的字典
        """
        try:
            # BTC-USDC -> UBTC/USDC
            trading_pair = self._get_trading_pair_name(pair)
            
            if self.debug:
                print(f"🔧 [HYPERLIQUID GET PRICE] Getting price for pair: {pair} -> {trading_pair}")

            # 使用 all_mids() 获取所有中间价
            all_mids = self.info.all_mids()
            
            if self.debug:
                print(f"🔍 [HYPERLIQUID ALL MIDS] Retrieved {len(all_mids)} trading pairs")
                # 只显示部分数据，避免输出过多
                if trading_pair in all_mids:
                    print(f"🔍 [HYPERLIQUID GET PRICE] Found {trading_pair}: {all_mids[trading_pair]}")

            # 检查交易对是否存在并获取价格
            # 对于 UBTC/USDC，尝试多种可能的格式
            possible_formats = [trading_pair]
            if trading_pair == "UBTC/USDC":
                possible_formats.extend(["@142", "UBTC"])
            
            price = None
            found_format = None
            
            for format_name in possible_formats:
                if format_name in all_mids:
                    price = float(all_mids[format_name])
                    found_format = format_name
                    break
            
            if price is not None:
                if self.debug:
                    print(f"✅ [HYPERLIQUID GET PRICE SUCCESS] {found_format} price: {price}")

                return {
                    "success": True,
                    "symbol": trading_pair,
                    "price": price,
                    "exchange": "hyperliquid",
                    "trading_pair": pair,  # 返回原始输入的交易对
                    "hyperliquid_pair": found_format
                }

            # 如果找不到交易对
            error_msg = f"Trading pair {trading_pair} not found in all_mids"
            if self.debug:
                print(f"❌ [HYPERLIQUID GET PRICE ERROR] {error_msg}")
                # 显示可用的交易对（仅前几个）
                available_pairs = list(all_mids.keys())[:5]
                print(f"❌ [HYPERLIQUID GET PRICE ERROR] Available pairs (first 5): {available_pairs}")

            return {
                "success": False,
                "error": error_msg,
                "exchange": "hyperliquid",
                "trading_pair": pair,
                "hyperliquid_pair": trading_pair
            }

        except Exception as e:
            error_msg = f"Error getting ticker price: {e}"
            if self.debug:
                print(f"❌ [HYPERLIQUID GET PRICE ERROR] {error_msg}")
                import traceback
                print(f"❌ [HYPERLIQUID GET PRICE ERROR] Traceback: {traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "exchange": "hyperliquid",
                "trading_pair": pair
            }

    def place_market_order(self, side: str, pair: str, amount: float) -> Dict[str, Any]:
        """
        下市价单
        
        Args:
            side: 交易方向，"BUY" 或 "SELL"
            pair: 交易对，如 "BTC-USDC"
            amount: 数量（BTC 数量）
            
        Returns:
            包含订单信息的字典
        """
        try:
            # BTC-USDC -> UBTC/USDC
            trading_pair = self._get_trading_pair_name(pair)
            is_buy = (side.upper() == "BUY")
            
            if self.debug:
                print(f"🔧 [HYPERLIQUID ORDER] Original pair: {pair}, Trading pair: {trading_pair}")
                print(f"🔧 [HYPERLIQUID ORDER] Side: {side} -> {is_buy}, Amount: {amount}")

            # 使用 exchange.market_open 下现货市价单
            order_result = self.exchange.market_open(
                trading_pair,             # "BTC" - 币种名称
                is_buy,                   # True/False - 买卖方向
                amount,                   # 数量
                px=None,                  # 价格限制（None表示使用当前市价）
                slippage=0.05             # 允许滑点（5%）
            )
            
            if self.debug:
                print(f"🔍 [HYPERLIQUID ORDER RESPONSE] {json.dumps(order_result, indent=2)}")

            # 解析订单结果
            if order_result.get("status") == "ok":
                # 处理成功响应
                response_data = order_result.get("response", {}).get("data", {})
                statuses = response_data.get("statuses", [])
                
                order_id = None
                status = "SUBMITTED"
                
                # 尝试从 statuses 中提取订单信息
                if statuses and len(statuses) > 0:
                    first_status = statuses[0]
                    
                    # 检查是否有错误
                    if 'error' in first_status:
                        error_msg = first_status['error']
                        if self.debug:
                            print(f"❌ [HYPERLIQUID ORDER ERROR] Order failed: {error_msg}")
                        
                        return {
                            "success": False,
                            "error": error_msg,
                            "exchange": "hyperliquid",
                            "trading_pair": pair,
                            "hyperliquid_pair": trading_pair,
                            "amount": amount,
                            "side": side,
                            "raw_response": order_result
                        }
                    # 检查是否有填充信息
                    elif 'filled' in first_status:
                        filled_info = first_status['filled']
                        order_id = filled_info.get('oid')
                        status = "FILLED"
                    elif 'resting' in first_status:
                        resting_info = first_status['resting']
                        order_id = resting_info.get('oid')
                        status = "OPEN"

                if self.debug:
                    print(f"✅ [HYPERLIQUID ORDER SUCCESS] Order placed successfully")
                    print(f"✅ [HYPERLIQUID ORDER SUCCESS] Order ID: {order_id}, Status: {status}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "client_order_id": "",
                    "symbol": trading_pair,
                    "side": side,
                    "type": "MARKET",
                    "quantity": str(amount),
                    "status": status,
                    "exchange": "hyperliquid",
                    "trading_pair": pair,
                    "hyperliquid_pair": trading_pair,
                    "raw_response": order_result
                }
            else:
                # 处理错误响应
                response = order_result.get("response", "Unknown error")
                if isinstance(response, str):
                    error_msg = response
                else:
                    error_msg = response.get("data", {}).get("error", "Unknown error")
                
                if self.debug:
                    print(f"❌ [HYPERLIQUID ORDER ERROR] Order failed: {error_msg}")

                return {
                    "success": False,
                    "error": error_msg,
                    "exchange": "hyperliquid",
                    "trading_pair": pair,
                    "hyperliquid_pair": trading_pair,
                    "amount": amount,
                    "side": side,
                    "raw_response": order_result
                }

        except Exception as e:
            error_msg = f"Error placing market order: {e}"
            if self.debug:
                print(f"❌ [HYPERLIQUID ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [HYPERLIQUID ORDER ERROR] Traceback: {traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "exchange": "hyperliquid",
                "trading_pair": pair,
                "amount": amount,
                "side": side
            }

    def get_order_details(self, order_id: str, pair: str) -> Dict[str, Any]:
        """
        查询订单详情
        
        Args:
            order_id: 订单 ID
            pair: 交易对，如 "BTC-USDC"
            
        Returns:
            包含订单详情的字典
        """
        try:
            trading_pair = self._get_trading_pair_name(pair)
            
            if self.debug:
                print(f"🔧 [HYPERLIQUID GET ORDER] Order ID: {order_id}, Pair: {pair} -> {trading_pair}")

            # 使用 info.query_order_by_oid 查询订单
            order_details = self.info.query_order_by_oid(
                user=self.user_address,
                oid=order_id
            )
            
            if self.debug:
                print(f"🔍 [HYPERLIQUID GET ORDER RESPONSE] {json.dumps(order_details, indent=2)}")

            if order_details and 'order' in order_details:
                order_info = order_details['order']
                
                # 解析订单信息
                side = "BUY" if order_info.get('side') == 'A' else "SELL"  # A = Ask (sell), B = Bid (buy)
                quantity = order_info.get('sz', '0')
                status = self._parse_order_status(order_info.get('orderStatus', ''))
                
                if self.debug:
                    print(f"✅ [HYPERLIQUID GET ORDER SUCCESS] Order details retrieved")

                return {
                    "success": True,
                    "order_id": order_id,
                    "client_order_id": "",
                    "symbol": trading_pair,
                    "side": side,
                    "type": "MARKET",
                    "quantity": quantity,
                    "executed_quantity": order_info.get('filledSz', '0'),
                    "price": "",
                    "average_price": order_info.get('avgPx', ''),
                    "status": status,
                    "create_time": order_info.get('timestamp', ''),
                    "update_time": order_info.get('timestamp', ''),
                    "exchange": "hyperliquid",
                    "trading_pair": pair,
                    "hyperliquid_pair": trading_pair,
                    "raw_response": order_details
                }
            else:
                error_msg = "Order not found or invalid response"
                if self.debug:
                    print(f"❌ [HYPERLIQUID GET ORDER ERROR] {error_msg}")

                return {
                    "success": False,
                    "error": error_msg,
                    "order_id": order_id,
                    "exchange": "hyperliquid",
                    "trading_pair": pair,
                    "hyperliquid_pair": trading_pair
                }

        except Exception as e:
            error_msg = f"Error getting order details: {e}"
            if self.debug:
                print(f"❌ [HYPERLIQUID GET ORDER ERROR] {error_msg}")
                import traceback
                print(f"❌ [HYPERLIQUID GET ORDER ERROR] Traceback: {traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "order_id": order_id,
                "exchange": "hyperliquid",
                "trading_pair": pair
            }

    def _parse_order_status(self, status: str) -> str:
        """
        解析订单状态
        
        Args:
            status: Hyperliquid 原始状态
            
        Returns:
            标准化状态
        """
        status_mapping = {
            'open': 'OPEN',
            'filled': 'FILLED',
            'canceled': 'CANCELED',
            'rejected': 'REJECTED'
        }
        
        return status_mapping.get(status.lower(), 'UNKNOWN')


def main():
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(description='Hyperliquid 现货直连交易 API')
    parser.add_argument('--action', type=str, required=True, choices=['balance', 'order'], 
                      help='操作类型: balance(查询余额) 或 order(下单)')
    parser.add_argument('--pair', type=str, default='BTC-USDC', 
                      help='交易对 (默认: BTC-USDC)')
    parser.add_argument('--amount', type=float, 
                      help='交易数量 (BTC数量)')
    parser.add_argument('--side', type=str, choices=['buy', 'sell'], 
                      help='交易方向: buy 或 sell')
    parser.add_argument('--debug', action='store_true', 
                      help='开启调试模式')

    args = parser.parse_args()

    try:
        # 初始化 API
        api = HyperliquidSpotDirectAPI(debug=args.debug)

        if args.action == 'balance':
            print("=== 查询账户余额 ===")
            result = api.get_account_balances(show_zero_balances=False, min_balance=0.0)
            print(json.dumps(result, indent=2))

        elif args.action == 'order':
            if not args.amount or not args.side:
                print("❌ 下单需要指定 --amount 和 --side 参数")
                sys.exit(1)

            print(f"=== 执行{args.side.upper()}单 ===")
            print(f"交易对: {args.pair}")
            print(f"数量: {args.amount}")
            print(f"方向: {args.side.upper()}")
            
            # 下单
            result = api.place_market_order(args.side.upper(), args.pair, args.amount)
            print("下单结果:")
            print(json.dumps(result, indent=2))
            
            # 如果下单成功，查询订单详情
            if result.get('success') and result.get('order_id'):
                print(f"\n=== 订单详情 (Order ID: {result['order_id']}) ===")
                order_details = api.get_order_details(result['order_id'], args.pair)
                print(json.dumps(order_details, indent=2))

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()