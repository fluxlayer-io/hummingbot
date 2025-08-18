#!/usr/bin/env python3
"""
FluxLayer 自动资产平衡脚本

自动检查各交易所的 BTC 和 USDC 余额，如果不足 10U 则自动购买
使用 USDT 作为购买货币，确保所有交易所都有足够的资产用于交易
"""
import asyncio
import os
import sys
import time
import argparse
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(project_root)

from test_balance_query import BalanceQueryService
from hummingbot.fluxlayer_api.order_manager import OrderManager, get_order_manager
from hummingbot.fluxlayer_api.exchange_constants import EXCHANGES, get_supported_exchanges, is_stablecoin


class AutoOrderManager:
    """自动下单管理器 - 确保各交易所有足够的 BTC 和 USDC 余额"""
    
    def __init__(self, target_value_usd: float = 10.0):
        self.target_value_usd = target_value_usd
        self.balance_service = BalanceQueryService()
        self.order_manager = get_order_manager()
        
        # 目标资产配置
        self.target_assets = ["BTC", "USDC"]
        
        # 交易对映射 - 每个交易所购买特定资产的交易对
        self.trading_pairs = {
            "binance": {"BTC": "BTC-USDT", "USDC": "USDC-USDT"},
            "bybit": {"BTC": "BTC-USDT", "USDC": "USDC-USDT"},  
            "okx": {"BTC": "BTC-USDT", "USDC": "USDC-USDT"},
            "hyperliquid": {"BTC": "UBTC-USDC", "USDC": None}  # Hyperliquid 原生支持 USDC
        }
        
        # 执行统计
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "total_cost_usdt": 0,
            "exchanges_processed": 0
        }
        
    async def check_all_exchanges_assets(self) -> Dict[str, Any]:
        """
        检查所有交易所的 BTC 和 USDC 资产状态
        
        返回:
            dict: 包含各交易所资产状态的详细信息
        """
        print("🔍 检查所有交易所的 BTC 和 USDC 资产状态...")
        
        # 获取所有交易所余额
        all_balances = await self.balance_service.get_all_exchanges_balances()
        
        asset_status = {}
        total_usdt_available = 0
        
        # 获取实时 BTC 价格（用于所有交易所的计算）
        current_btc_price = None
        print("📈 获取实时 BTC 价格用于资产价值计算...")
        
        # 尝试从可用的交易所获取 BTC 价格
        for exchange_name in ["okx", "bybit", "binance"]:  # 优先使用这些交易所
            if exchange_name in all_balances["individual_exchanges"]:
                exchange_result = all_balances["individual_exchanges"][exchange_name]
                if exchange_result["status"] == "success":
                    try:
                        price = await self.get_market_price(exchange_name, "BTC-USDT")
                        if price and price > 0:
                            current_btc_price = price
                            print(f"✅ 使用 {exchange_name} 的 BTC 价格进行计算: ${current_btc_price:.2f}")
                            break
                    except Exception as e:
                        print(f"⚠️ 从 {exchange_name} 获取 BTC 价格失败: {e}")
                        continue
        
        # 如果无法获取实时价格，显示错误并停止
        if not current_btc_price:
            print("❌ 无法获取 BTC 实时价格，无法准确计算资产价值！")
            print("请检查网络连接和交易所 API 状态")
            return {
                "asset_status": {},
                "total_usdt_available": 0,
                "total_needed_usdt": 0,
                "funding_sufficient": False,
                "error": "Failed to get BTC price"
            }
        
        # 分析每个交易所的资产状态
        for exchange_name, exchange_result in all_balances["individual_exchanges"].items():
            if exchange_result["status"] == "success":
                balances = exchange_result["balances"]
                
                # 获取当前 BTC 和 USDC 余额
                btc_balance = balances.get("BTC", {}).get("total", 0)
                usdc_balance = balances.get("USDC", {}).get("total", 0)
                usdt_balance = balances.get("USDT", {}).get("total", 0)
                
                # 对 Hyperliquid 特殊处理 UBTC
                if exchange_name == "hyperliquid":
                    ubtc_balance = balances.get("UBTC", {}).get("total", 0)
                    btc_balance = ubtc_balance  # UBTC 等价于 BTC
                
                # 计算资产价值 (使用实时获取的 BTC 价格)
                btc_value_usd = btc_balance * current_btc_price  # 实时BTC价格
                usdc_value_usd = usdc_balance * 1                # USDC 稳定币价格
                
                # 计算需要补充的数量
                btc_shortage = max(0, self.target_value_usd - btc_value_usd)
                usdc_shortage = max(0, self.target_value_usd - usdc_value_usd)
                
                asset_status[exchange_name] = {
                    "current_balances": {
                        "BTC": btc_balance,
                        "USDC": usdc_balance, 
                        "USDT": usdt_balance
                    },
                    "current_values_usd": {
                        "BTC": btc_value_usd,
                        "USDC": usdc_value_usd
                    },
                    "shortages_usd": {
                        "BTC": btc_shortage,
                        "USDC": usdc_shortage
                    },
                    "needs_btc": btc_shortage > 0,
                    "needs_usdc": usdc_shortage > 0,
                    "total_needed_usdt": btc_shortage + usdc_shortage,
                    "status": "success"
                }
                
                total_usdt_available += usdt_balance
                
                # 打印交易所状态
                print(f"\n🏢 {exchange_name.upper()}")
                
                btc_status = "✅" if btc_value_usd >= self.target_value_usd else f"❌ 需要补充 ${btc_shortage:.2f}"
                usdc_status = "✅" if usdc_value_usd >= self.target_value_usd else f"❌ 需要补充 ${usdc_shortage:.2f}"
                
                print(f"   💰 BTC: {btc_balance:.6f} (${btc_value_usd:.2f}) {btc_status}")
                print(f"   💰 USDC: {usdc_balance:.6f} (${usdc_value_usd:.2f}) {usdc_status}")  
                print(f"   💰 USDT: {usdt_balance:.6f} (可用于购买)")
                
            else:
                asset_status[exchange_name] = {
                    "status": "error",
                    "error": exchange_result.get("error_message", "Unknown error"),
                    "needs_btc": False,
                    "needs_usdc": False,
                    "total_needed_usdt": 0
                }
                print(f"\n🏢 {exchange_name.upper()}")
                print(f"   ❌ 错误: {exchange_result.get('error_message', 'Unknown error')}")
        
        # 计算总需求
        total_needed_usdt = sum(
            status.get("total_needed_usdt", 0) 
            for status in asset_status.values() 
            if status.get("status") == "success"
        )
        
        print(f"\n📊 资产状态汇总:")
        print(f"   总 USDT 可用: ${total_usdt_available:.2f}")
        print(f"   总 USDT 需求: ${total_needed_usdt:.2f}")
        print(f"   资金状态: {'✅ 充足' if total_usdt_available >= total_needed_usdt else '❌ 不足'}")
        
        return {
            "asset_status": asset_status,
            "total_usdt_available": total_usdt_available,
            "total_needed_usdt": total_needed_usdt,
            "funding_sufficient": total_usdt_available >= total_needed_usdt
        }
    
    async def get_market_price(self, exchange_name: str, trading_pair: str) -> Optional[float]:
        """
        获取指定交易对的当前市价（使用修复后的 order_manager）
        
        参数:
            exchange_name: 交易所名称
            trading_pair: 交易对
            
        返回:
            float: 当前价格，如果获取失败返回 None
        """
        try:
            # USDC 稳定币价格
            if "USDC" in trading_pair:
                print(f"   💵 使用 USDC 稳定币价格: $1.00")
                return 1.0
            
            # 使用修复后的 order_manager 获取价格（现在对 Binance 有容错处理）
            try:
                connector = await self.order_manager._get_or_create_connector(
                    exchange_name, trading_pair, wait_for_orderbook=False
                )
                
                if hasattr(connector, '_get_last_traded_price'):
                    price = await connector._get_last_traded_price(trading_pair)
                    if price and price > 0:
                        if self._validate_price(trading_pair, float(price)):
                            print(f"   📈 获取 {exchange_name} {trading_pair} 实时价格: ${price:.2f}")
                            return float(price)
                        else:
                            print(f"   ⚠️ {exchange_name} {trading_pair} 价格异常: ${price:.2f}")
                            return None
                            
            except Exception as connector_error:
                print(f"   ⚠️ 无法从 {exchange_name} 获取 {trading_pair} 实时价格: {connector_error}")
                return None
            
            # 如果所有方法都失败，返回 None
            print(f"   ❌ 无法从 {exchange_name} 获取 {trading_pair} 价格")
            return None
                
        except Exception as e:
            print(f"⚠️ 获取 {exchange_name} {trading_pair} 价格时发生异常: {e}")
            return None
    
    def _validate_price(self, trading_pair: str, price: float) -> bool:
        """
        验证价格是否在合理范围内
        
        参数:
            trading_pair: 交易对
            price: 价格
            
        返回:
            bool: 价格是否合理
        """
        if "BTC" in trading_pair or "UBTC" in trading_pair:
            # BTC 价格应该在 50,000 到 200,000 美元之间
            return 50000 <= price <= 200000
        elif "USDC" in trading_pair:
            # USDC 价格应该在 0.95 到 1.05 美元之间
            return 0.95 <= price <= 1.05
        else:
            # 其他代币价格应该大于 0
            return price > 0
    
    def calculate_purchase_amount(self, target_value_usd: float, current_price: float) -> float:
        """
        计算需要购买的数量
        
        参数:
            target_value_usd: 目标价值（美元）
            current_price: 当前价格
            
        返回:
            float: 需要购买的数量
        """
        return target_value_usd / current_price
    
    async def execute_purchase_orders(self, asset_analysis: Dict[str, Any], check_only: bool = False) -> Dict[str, Any]:
        """
        执行购买订单
        
        参数:
            asset_analysis: 资产分析结果
            check_only: 仅检查模式，不实际下单
            
        返回:
            dict: 执行结果
        """
        if not asset_analysis["funding_sufficient"]:
            print(f"❌ USDT 余额不足！需要 ${asset_analysis['total_needed_usdt']:.2f}，但只有 ${asset_analysis['total_usdt_available']:.2f}")
            print("\n💰 详细余额信息:")
            
            # 显示所有交易所的完整余额信息
            for exchange_name, status in asset_analysis["asset_status"].items():
                if status["status"] == "success":
                    current_balances = status["current_balances"]
                    print(f"\n🏢 {exchange_name.upper()}:")
                    for token, balance in current_balances.items():
                        print(f"   💰 {token}: {balance:.6f}")
            
            return {
                "success": False,
                "error": "Insufficient USDT balance",
                "orders": []
            }
        
        if check_only:
            print(f"\n✅ 检查模式：资金充足，可以执行购买")
            print(f"   需要购买总价值: ${asset_analysis['total_needed_usdt']:.2f}")
            return {
                "success": True,
                "check_only": True,
                "orders": []
            }
        
        print(f"\n🚀 开始执行购买订单...")
        
        executed_orders = []
        
        # 初始化订单管理器
        await self.order_manager.initialize_all_connectors()
        
        for exchange_name, status in asset_analysis["asset_status"].items():
            if status["status"] != "success":
                continue
                
            if not (status["needs_btc"] or status["needs_usdc"]):
                print(f"✅ {exchange_name} 资产充足，跳过")
                continue
            
            print(f"\n🔧 处理 {exchange_name} 交易所...")
            
            self.execution_stats["exchanges_processed"] += 1
            
            # 购买 BTC
            if status["needs_btc"]:
                await self._execute_single_purchase(
                    exchange_name, "BTC", executed_orders
                )
            
            # 购买 USDC (Hyperliquid 跳过，因为它原生支持 USDC)
            if status["needs_usdc"] and exchange_name != "hyperliquid":
                await self._execute_single_purchase(
                    exchange_name, "USDC", executed_orders
                )
        
        # 执行统计
        successful_orders = sum(1 for order in executed_orders if order.get("success"))
        failed_orders = len(executed_orders) - successful_orders
        
        self.execution_stats.update({
            "successful_orders": successful_orders,
            "failed_orders": failed_orders,
            "total_orders": len(executed_orders)
        })
        
        print(f"\n📊 执行完成统计:")
        print(f"   总订单数: {len(executed_orders)}")
        print(f"   成功订单: {successful_orders}")
        print(f"   失败订单: {failed_orders}")
        print(f"   处理交易所: {self.execution_stats['exchanges_processed']}")
        
        return {
            "success": failed_orders == 0,
            "total_orders": len(executed_orders),
            "successful_orders": successful_orders,  
            "failed_orders": failed_orders,
            "orders": executed_orders
        }
    
    async def _execute_single_purchase(
        self, 
        exchange_name: str, 
        asset: str, 
        executed_orders: List[Dict]
    ):
        """
        执行单个资产的购买订单
        
        参数:
            exchange_name: 交易所名称
            asset: 资产名称 (BTC/USDC)
            executed_orders: 执行订单列表
        """
        try:
            # 获取交易对
            trading_pair = self.trading_pairs[exchange_name].get(asset)
            if not trading_pair:
                print(f"   ⚠️ {exchange_name} 不支持购买 {asset}")
                return
            
            # 获取市价
            current_price = await self.get_market_price(exchange_name, trading_pair)
            if not current_price:
                print(f"   ❌ 无法获取 {exchange_name} {trading_pair} 价格")
                return
            
            # 计算购买数量（使用固定的目标金额，不是缺口金额）
            purchase_amount = self.calculate_purchase_amount(self.target_value_usd, current_price)
            
            print(f"   💰 购买 {asset}: {purchase_amount:.6f} {asset} (${self.target_value_usd:.2f}) 在 {trading_pair}")
            
            # 执行市价买单
            order_result = await self.order_manager.place_order(
                connector_name=exchange_name,
                trading_pair=trading_pair,
                amount=purchase_amount,
                is_buy=True,
                order_type="MARKET"
            )
            
            # 记录订单结果
            order_info = {
                "exchange": exchange_name,
                "asset": asset,
                "trading_pair": trading_pair,
                "amount": purchase_amount,
                "target_value_usd": self.target_value_usd,
                "price": current_price,
                "success": order_result.get("success", False),
                "order_id": order_result.get("order_id"),
                "error": order_result.get("error"),
                "timestamp": time.time()
            }
            
            executed_orders.append(order_info)
            
            if order_result.get("success"):
                order_id = order_result.get('order_id')
                print(f"   ✅ 订单提交成功: {order_id}")
                self.execution_stats["total_cost_usdt"] += self.target_value_usd
                
                # 简单的订单状态检查（等待3秒后检查一次）
                try:
                    print(f"   ⏳ 等待3秒后检查订单状态...")
                    await asyncio.sleep(3)
                    
                    order_status = await self.order_manager.get_order_status(
                        connector_name=exchange_name,
                        order_id=order_id,
                        trading_pair=trading_pair,
                        max_retries=2
                    )
                    
                    if order_status.get("success"):
                        status = order_status.get("status", "UNKNOWN")
                        filled_amount = order_status.get("filled_amount", 0)
                        print(f"   📋 订单状态: {status}, 已成交: {filled_amount:.6f} {asset}")
                    else:
                        print(f"   ⚠️ 无法获取订单状态: {order_status.get('error')}")
                        
                except Exception as status_error:
                    print(f"   ⚠️ 订单状态检查失败: {status_error}")
                    
            else:
                print(f"   ❌ 订单失败: {order_result.get('error')}")
            
        except Exception as e:
            print(f"   ❌ 执行购买 {asset} 时发生错误: {e}")
            
            # 记录错误订单
            error_order = {
                "exchange": exchange_name,
                "asset": asset,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
            executed_orders.append(error_order)
    
    async def run_auto_balance(self, check_only: bool = False, target_exchanges: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行自动资产平衡
        
        参数:
            check_only: 仅检查模式，不实际下单
            target_exchanges: 目标交易所列表，None表示所有支持的交易所
            
        返回:
            dict: 执行结果
        """
        print("🎯 FluxLayer 自动资产平衡系统")
        print("=" * 50)
        print(f"目标资产配置: BTC ≥ ${self.target_value_usd}, USDC ≥ ${self.target_value_usd}")
        print(f"运行模式: {'仅检查' if check_only else '自动执行'}")
        
        if target_exchanges:
            print(f"目标交易所: {', '.join(target_exchanges)}")
            # 过滤交易所（这里简化处理，实际可以在 balance_service 中实现过滤）
        
        try:
            # 1. 检查资产状态
            asset_analysis = await self.check_all_exchanges_assets()
            
            # 2. 如果需要，执行购买订单
            execution_result = await self.execute_purchase_orders(asset_analysis, check_only)
            
            # 3. 最终报告
            print(f"\n🎊 自动资产平衡完成!")
            
            return {
                "success": execution_result.get("success", True),
                "asset_analysis": asset_analysis,
                "execution_result": execution_result,
                "execution_stats": self.execution_stats
            }
            
        except Exception as e:
            print(f"❌ 自动资产平衡过程中发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            # 清理资源
            await self.balance_service.cleanup()
    

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FluxLayer 自动资产平衡工具")
    parser.add_argument("--target-value", type=float, default=10.0, help="目标资产价值（美元），默认 10")
    parser.add_argument("--check-only", action="store_true", help="仅检查模式，不实际下单")
    parser.add_argument("--exchanges", type=str, help="指定交易所，用逗号分隔，如: binance,bybit")
    
    args = parser.parse_args()
    
    # 解析交易所列表
    target_exchanges = None
    if args.exchanges:
        target_exchanges = [ex.strip() for ex in args.exchanges.split(",")]
    
    # 创建自动下单管理器
    auto_manager = AutoOrderManager(target_value_usd=args.target_value)
    
    # 运行自动平衡
    result = await auto_manager.run_auto_balance(
        check_only=args.check_only,
        target_exchanges=target_exchanges
    )
    
    # 显示最终结果
    if result["success"]:
        print(f"✅ 自动资产平衡{'检查' if args.check_only else '执行'}成功完成")
    else:
        print(f"❌ 自动资产平衡{'检查' if args.check_only else '执行'}失败: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断操作")
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)