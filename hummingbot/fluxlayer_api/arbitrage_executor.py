import asyncio
import json
import logging
import os
from base64 import b64decode
from typing import Optional

import base58
import bitcoin
import nacl.encoding
import nacl.signing
import requests
from bitcoin.core import COIN, CMutableTransaction, CMutableTxIn, CMutableTxOut, COutPoint, b2x, lx
from bitcoin.core.script import OP_0, SIGHASH_ALL, SIGVERSION_WITNESS_V0, CScript, SignatureHash
from bitcoin.wallet import CBitcoinAddress, CBitcoinSecret, P2PKHBitcoinAddress
from solana.keypair import Keypair
from solana.publickey import PublicKey
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import TransferCheckedParams, transfer_checked

from hummingbot.fluxlayer_api.mpc_client import MPCClient

# 基于 const.ts 的代币地址映射
TOKEN_MAPPINGS = {
    # Solana 主网代币
    "SOL": {
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "SOL": None  # 原生SOL
    },
    
    # Bitcoin 支持原生BTC
    "BTC": {
        "BTC": None  # 原生BTC
    },
    
    # 以太坊和其他EVM链的代币映射（来自const.ts）
    "ETH": {
        "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
        "WBTC": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "ETH": None  # 原生ETH
    },
    
    "BSC_BNB": {
        "USDC": "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
        "USDT": "0x55d398326f99059ff775485246999027b3197955",
        "BTCB": "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "BNB": None  # 原生BNB
    },
    
    "ARBITRUM_ETH": {
        "USDC": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
        "USDT": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "ETH": None  # 原生ETH
    }
}

# 简单 Logger 配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fluxlayer Arbitrage Executor")

# ========== 简单私钥获取函数 ==========

def get_private_key_from_env(chain: str) -> Optional[str]:
    """从环境变量获取私钥"""
    env_vars = {
        "SOL": ["SOL_PRIVATE_KEY", "SOLANA_PRIVATE_KEY"],
        "BTC": ["BTC_PRIVATE_KEY", "BITCOIN_PRIVATE_KEY"],
        "XTN": ["XTN_PRIVATE_KEY", "XTON_PRIVATE_KEY", "BTC_PRIVATE_KEY", "BITCOIN_PRIVATE_KEY"],  # XTN 可以使用 BTC 私钥
        "ETH": ["ETH_PRIVATE_KEY", "ETHEREUM_PRIVATE_KEY"],
    }
    
    possible_vars = env_vars.get(chain, [f"{chain}_PRIVATE_KEY"])
    
    for var_name in possible_vars:
        private_key = os.getenv(var_name)
        if private_key and private_key.strip():
            logger.info(f"🔑 从环境变量 {var_name} 获取 {chain} 私钥")
            return private_key.strip()
    
    logger.error(f"❌ 未找到 {chain} 私钥，请设置环境变量：{' 或 '.join(possible_vars)}")
    return None

# Mock FluxLayer Exchange Metadata
class MockMetadata:
    def __init__(self):
        self.trading_pairs = {
            "XTN-USDC": MockPairMeta()
        }
        self.target_amount = 0.001
        self.source_amount = 0.0001
        self.is_buy = False

class MockPairMeta:
    def __init__(self):
        self.source_chain = "SOL"
        self.target_chain = "XTN"
        self.source_token = "USDC"
        self.target_token = "XTN"

class MockFluxLayerExchange:
    def __init__(self):
        self.metadata = MockMetadata()

class ChainAdapter:
    def get_address_from_private_key(self, private_key: str) -> str:
        raise NotImplementedError
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str) -> str:
        raise NotImplementedError
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        raise NotImplementedError

class SolanaAdapter(ChainAdapter):
    
    def get_address_from_private_key(self, private_key: str) -> str:
        if not private_key:
            raise ValueError("Private key not provided")

        try:
            # 支持 Base58, Base64 和 JSON 数组格式
            if private_key.startswith("[") and private_key.endswith("]"):
                secret_key = bytes(json.loads(private_key))
            elif len(private_key) in [87, 88] and all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in private_key):
                secret_key = base58.b58decode(private_key)
            else:
                secret_key = b64decode(private_key)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid private key format: {e}")

        if len(secret_key) not in [32, 64]:
            if len(secret_key) == 32:
                keypair = Keypair.from_seed(secret_key)
                return str(keypair.public_key)
            else:
                raise ValueError(f"Invalid private key length: {len(secret_key)} bytes, expected 32 or 64")

        keypair = Keypair.from_secret_key(secret_key)
        return str(keypair.public_key)
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        try:
            # 解析私钥
            if private_key.startswith("[") and private_key.endswith("]"):
                secret = bytes(json.loads(private_key))
            elif len(private_key) in [87, 88] and all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in private_key):
                secret = base58.b58decode(private_key)
            else:
                secret = b64decode(private_key)
            
            if len(secret) not in [32, 64]:
                if len(secret) == 32:
                    keypair = Keypair.from_seed(secret)
                    secret = keypair.secret_key
                else:
                    raise ValueError(f"Invalid private key length: {len(secret)} bytes, expected 32 or 64")
            
            # 使用前32字节进行签名
            signing_secret = secret[:32] if len(secret) == 64 else secret
            signing_key = nacl.signing.SigningKey(signing_secret)
            signed = signing_key.sign(message)
            return signed.signature.hex()
            
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid private key format: {e}")
        except Exception as e:
            raise Exception(f"Signing failed: {e}")
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str = "SOL") -> str:
        from solana.keypair import Keypair
        from solana.publickey import PublicKey
        from solana.rpc.async_api import AsyncClient
        from solana.rpc.types import TxOpts
        from solana.system_program import TransferParams, transfer
        from solana.transaction import Transaction

        # 使用官方主网端点
        rpc_endpoints = [
            "https://api.mainnet-beta.solana.com"
        ]
        
        client = None
        for endpoint in rpc_endpoints:
            try:
                client = AsyncClient(endpoint)
                logger.info(f"🔗 使用RPC端点: {endpoint}")
                break
            except Exception as e:
                logger.warning(f"⚠️ RPC端点 {endpoint} 连接失败: {e}")
                continue
        
        if not client:
            raise Exception("所有RPC端点都连接失败")
        
        # 使用全局代币映射
        TOKEN_MINTS = TOKEN_MAPPINGS.get("SOL", {})
        
        try:
            logger.info(f"🔄 转账: {amount} {token} 从私钥钱包到 {to_address}")
            
            # 解析私钥
            if len(private_key) in [87, 88]:
                secret = base58.b58decode(private_key)
            else:
                raise ValueError("私钥格式不正确，请使用Base58格式")
            
            sender = Keypair.from_secret_key(secret)
            recipient = PublicKey(to_address)
            logger.info(f"💳 发送方地址: {sender.public_key}")
            logger.info(f"🏦 接收方地址: {recipient}")
            
            # 获取latest blockhash
            import httpx
            
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getLatestBlockhash",
                    "params": [{"commitment": "processed"}]
                }
                
                rpc_url = client._provider.endpoint_uri
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        rpc_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    result = response.json()
                    logger.info(f"📡 Latest blockhash response: {result}")
                    
                    if 'result' in result and result['result'] is not None:
                        value = result['result']['value']
                        if 'blockhash' in value:
                            blockhash = value['blockhash']
                            logger.info(f"📡 获取到最新blockhash: {blockhash}")
                        else:
                            raise Exception(f"响应中没有blockhash: {value}")
                    else:
                        raise Exception(f"getLatestBlockhash请求失败: {result}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to get latest blockhash: {e}")
                raise
            
            # 创建交易
            tx = Transaction(recent_blockhash=blockhash)
            
            if token == "SOL":
                # 原生SOL转账
                amount_lamports = int(float(amount) * 1_000_000_000)
                logger.info(f"💰 转账金额: {amount} SOL ({amount_lamports} lamports)")
                
                # 检查余额
                balance_resp = await client.get_balance(sender.public_key)
                if 'result' in balance_resp and balance_resp['result'] is not None:
                    balance = balance_resp['result']['value']
                    logger.info(f"💵 发送方余额: {balance / 1_000_000_000:.6f} SOL")
                else:
                    raise Exception(f"Failed to get balance: {balance_resp}")
                
                if balance < amount_lamports:
                    raise ValueError(f"余额不足: 需要 {amount_lamports}, 可用 {balance}")
                
                tx.add(transfer(TransferParams(
                    from_pubkey=sender.public_key,
                    to_pubkey=recipient,
                    lamports=amount_lamports
                )))
                
            else:
                # SPL代币转账
                if token not in TOKEN_MINTS:
                    raise ValueError(f"不支持的代币: {token}")
                
                mint_address = PublicKey(TOKEN_MINTS[token])
                decimals = 6 if token in ["USDC", "USDT"] else 6
                token_amount = int(float(amount) * (10 ** decimals))
                
                logger.info(f"💰 转账金额: {amount} {token} ({token_amount} 基础单位)")
                
                # 获取代币账户
                from solana.rpc.types import TokenAccountOpts
                
                sender_token_accounts = await client.get_token_accounts_by_owner(
                    sender.public_key, 
                    TokenAccountOpts(mint=mint_address)
                )
                
                if not sender_token_accounts.get('result', {}).get('value', []):
                    raise ValueError(f"发送方没有{token}代币账户")
                
                sender_token_account = PublicKey(sender_token_accounts['result']['value'][0]['pubkey'])
                
                # 获取或创建接收方关联代币账户 (Associated Token Account)
                from spl.token.instructions import create_associated_token_account, get_associated_token_address

                # 计算关联代币账户地址
                recipient_token_account = get_associated_token_address(recipient, mint_address)
                logger.info(f"🔍 计算的接收方ATA地址: {recipient_token_account}")
                
                # 检查接收方是否已有代币账户
                recipient_token_accounts = await client.get_token_accounts_by_owner(
                    recipient, 
                    TokenAccountOpts(mint=mint_address)
                )
                
                # 如果接收方没有代币账户，创建关联代币账户指令
                if not recipient_token_accounts.get('result', {}).get('value', []):
                    logger.info(f"⚠️ 接收方没有{token}代币账户，将自动创建关联代币账户")
                    
                    # 添加创建关联代币账户的指令
                    create_ata_ix = create_associated_token_account(
                        payer=sender.public_key,  # 发送方支付创建费用
                        owner=recipient,          # 接收方拥有账户
                        mint=mint_address
                    )
                    tx.add(create_ata_ix)
                    logger.info(f"✅ 已添加创建ATA指令")
                else:
                    # 如果已有账户，使用现有的
                    existing_account = recipient_token_accounts['result']['value'][0]['pubkey']
                    recipient_token_account = PublicKey(existing_account)
                    logger.info(f"✅ 使用现有代币账户: {recipient_token_account}")
                
                # 检查余额
                balance_info = await client.get_token_account_balance(sender_token_account)
                if 'result' in balance_info and balance_info['result'] is not None:
                    balance = int(balance_info['result']['value']['amount'])
                    balance_ui = float(balance_info['result']['value']['uiAmountString'])
                    logger.info(f"💵 发送方{token}余额: {balance_ui} {token}")
                else:
                    raise Exception(f"Failed to get token balance: {balance_info}")
                
                if balance < token_amount:
                    raise ValueError(f"{token}余额不足: 需要 {token_amount}, 可用 {balance}")
                
                tx.add(transfer_checked(TransferCheckedParams(
                    program_id=TOKEN_PROGRAM_ID,
                    source=sender_token_account,
                    mint=mint_address,
                    dest=recipient_token_account,
                    owner=sender.public_key,
                    amount=token_amount,
                    decimals=decimals,
                    signers=[sender.public_key]
                )))
            
            # 签名并发送交易
            tx.sign(sender)
            logger.info("✍️ 交易已签名")
            
            try:
                # 发送已签名的交易，使用原始字节形式
                logger.info("📤 准备发送交易...")
                logger.info(f"🔍 交易对象: {tx}")
                logger.info(f"🔍 交易指令数量: {len(tx.instructions)}")
                logger.info(f"🔍 交易blockhash: {tx.recent_blockhash}")
                
                # 序列化交易为字节
                tx_bytes = tx.serialize()
                logger.info(f"📦 交易字节长度: {len(tx_bytes)}")
                
                # 使用 send_raw_transaction 发送已序列化的交易
                # 手动构建 sendTransaction RPC 请求
                import base64

                import httpx
                
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sendTransaction",
                    "params": [
                        base64.b64encode(tx_bytes).decode('utf-8'),  # 交易的base64编码
                        {
                            "skipPreflight": False,
                            "preflightCommitment": "confirmed",
                            "encoding": "base64"
                        }
                    ]
                }
                
                logger.info("📤 发送交易到RPC节点...")
                
                rpc_url = client._provider.endpoint_uri
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        rpc_url,
                        json=rpc_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    result = response.json()
                    logger.info(f"📡 交易发送响应: {result}")
                    
                    if 'result' in result and result['result'] is not None:
                        tx_signature = result['result']
                        logger.info(f"📤 交易已发送: {tx_signature}")
                    elif 'error' in result:
                        raise Exception(f"交易发送失败: {result['error']}")
                    else:
                        raise Exception(f"交易发送响应格式错误: {result}")
                        
            except Exception as send_error:
                logger.error(f"❌ 发送交易时出错: {send_error}")
                logger.error(f"❌ 错误类型: {type(send_error)}")
                import traceback
                traceback.print_exc()
                raise send_error
            
            # 简化确认过程 - 既然交易已经成功发送，我们等待几秒钟然后返回
            logger.info(f"⏳ 等待交易在链上确认: {tx_signature}")
            await asyncio.sleep(5)  # 等待5秒让交易上链
            
            logger.info(f"✅ 转账交易已发送: {tx_signature}")
            logger.info(f"🔍 可以在 Solana Explorer 查看交易: https://explorer.solana.com/tx/{tx_signature}")
            return tx_signature
            
        except Exception as e:
            logger.error(f"❌ 转账失败: {e}")
            return None
        finally:
            await client.close()

class BitcoinAdapter(ChainAdapter):
    """Bitcoin 区块链适配器 - 支持主网和测试网"""
    
    def __init__(self, network: str = "mainnet"):
        # 验证网络参数
        if network not in ["mainnet", "testnet", "testnet3", "testnet4"]:
            raise ValueError(f"不支持的网络: {network}，请使用 'mainnet', 'testnet', 'testnet3' 或 'testnet4'")
        
        self.network = network
        
        # 根据网络设置参数
        if network == "mainnet":
            bitcoin.SelectParams('mainnet')
            self.api_base = "https://blockstream.info/api"
            self.explorer_url = "https://blockstream.info/tx"
        elif network in ["testnet", "testnet3"]:
            bitcoin.SelectParams('testnet')
            self.api_base = "https://blockstream.info/testnet/api"
            self.explorer_url = "https://blockstream.info/testnet/tx"
        elif network == "testnet4":
            bitcoin.SelectParams('testnet')  # 使用 testnet 参数
            # testnet4 需要使用专门的 API
            self.api_base = "https://mempool.space/testnet4/api"
            self.explorer_url = "https://mempool.space/testnet4/tx"
        
        logger.info(f"🌐 Bitcoin适配器初始化: {network}")
        if network == "testnet4":
            logger.info(f"🔗 使用 mempool.space testnet4 API")
    
    def get_address_from_private_key(self, private_key: str, address_type: str = "auto") -> str:
        """从WIF格式私钥获取Bitcoin地址
        
        Args:
            private_key: WIF格式私钥
            address_type: 地址类型 - "auto", "p2pkh", "p2wpkh"
                - "auto": 自动选择（压缩公钥用SegWit，未压缩用Legacy）
                - "p2pkh": Legacy地址 (1xxx格式)
                - "p2wpkh": Native SegWit地址 (bc1xxx格式)
        """
        if not private_key:
            raise ValueError("Private key not provided")
        
        try:
            # 验证并转换WIF私钥 - 根据网络验证前缀
            expected_prefixes = self._get_wif_prefixes()
            
            if (len(private_key) in [51, 52] and 
                any(private_key.startswith(prefix) for prefix in expected_prefixes)):
                # 使用 python-bitcoinlib 的 CBitcoinSecret
                private_key_obj = CBitcoinSecret(private_key)
                
                # 判断公钥是否压缩
                is_compressed = len(private_key_obj.pub) == 33
                logger.info(f"🔑 私钥解析成功，公钥长度: {len(private_key_obj.pub)} ({'压缩' if is_compressed else '未压缩'})")
                
                # 确定实际使用的地址类型
                if address_type == "auto":
                    # 自动选择：压缩公钥用SegWit，未压缩用Legacy
                    actual_type = "p2wpkh" if is_compressed else "p2pkh"
                    logger.info(f"🎯 自动选择地址类型: {actual_type}")
                else:
                    actual_type = address_type
                    logger.info(f"🎯 指定地址类型: {actual_type}")
                
                # 生成对应类型的地址
                if actual_type == "p2pkh":
                    # Legacy P2PKH 地址 (1xxx)
                    address = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                    logger.info(f"✅ Legacy地址生成成功: {address}")
                    return str(address)
                    
                elif actual_type == "p2wpkh":
                    # Native SegWit P2WPKH 地址 (BIP84)
                    import hashlib

                    # 计算公钥哈希 (Hash160)
                    pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(private_key_obj.pub).digest()).digest()
                    
                    # 生成 bech32 地址
                    hrp = "bc" if self.network == "mainnet" else "tb"  # Human Readable Part
                    address_str = self._encode_bech32(hrp, 0, pubkey_hash)  # 0 = P2WPKH witness version
                    
                    if not address_str:
                        raise ValueError("生成P2WPKH地址失败")
                    
                    logger.info(f"✅ Native SegWit地址生成成功: {address_str}")
                    return address_str
                
                else:
                    raise ValueError(f"不支持的地址类型: {actual_type}")
                
            else:
                expected_str = " 或 ".join(expected_prefixes)
                raise ValueError(f"无效的WIF私钥格式，{self.network}网络应以 {expected_str} 开头")
                
        except Exception as e:
            raise ValueError(f"Bitcoin私钥解析失败: {e}")
    
    def _encode_bech32(self, hrp: str, witver: int, witprog: bytes) -> str:
        """Bech32 地址编码 (BIP173)"""
        try:
            # Bech32 字符集
            CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
            
            # Bech32 生成多项式
            GENERATOR = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
            
            def bech32_polymod(values):
                chk = 1
                for value in values:
                    top = chk >> 25
                    chk = (chk & 0x1ffffff) << 5 ^ value
                    for i in range(5):
                        chk ^= GENERATOR[i] if ((top >> i) & 1) else 0
                return chk
            
            def bech32_hrp_expand(hrp):
                return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
            
            def bech32_create_checksum(hrp, data):
                values = bech32_hrp_expand(hrp) + data
                polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
                return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
            
            def bech32_encode(hrp, data):
                combined = data + bech32_create_checksum(hrp, data)
                return hrp + '1' + ''.join([CHARSET[d] for d in combined])
            
            def convertbits(data, frombits, tobits, pad=True):
                acc = 0
                bits = 0
                ret = []
                maxv = (1 << tobits) - 1
                max_acc = (1 << (frombits + tobits - 1)) - 1
                for value in data:
                    if value < 0 or (value >> frombits):
                        return None
                    acc = ((acc << frombits) | value) & max_acc
                    bits += frombits
                    while bits >= tobits:
                        bits -= tobits
                        ret.append((acc >> bits) & maxv)
                if pad:
                    if bits:
                        ret.append((acc << (tobits - bits)) & maxv)
                elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
                    return None
                return ret
            
            # 转换 witness 程序
            ret = convertbits(witprog, 8, 5)
            if ret is None:
                return None
            
            # 创建数据载荷：witness版本 + 转换后的witness程序
            spec = [witver] + ret
            
            # 编码为 bech32
            return bech32_encode(hrp, spec)
            
        except Exception as e:
            logger.error(f"❌ Bech32编码失败: {e}")
            return None

    def _get_wif_prefixes(self) -> list:
        """获取当前网络的WIF私钥前缀"""
        if self.network == "mainnet":
            return ['5', 'K', 'L']  # 主网前缀
        else:  # testnet, testnet3, testnet4
            return ['9', 'c']  # 测试网前缀 (所有测试网都使用相同前缀)
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        """使用Bitcoin私钥签名消息"""
        try:
            private_key_obj = CBitcoinSecret(private_key)
            
            # 简化签名 - 直接对消息进行哈希并签名
            from bitcoin.core import Hash
            message_hash = Hash(message)
            
            # 使用私钥签名
            signature = private_key_obj.sign(message_hash)
            return signature.hex()
            
        except Exception as e:
            raise Exception(f"Bitcoin消息签名失败: {e}")
    
    async def _get_utxos(self, address: str):
        """获取地址的UTXO - 使用Blockstream API"""
        try:
            url = f"{self.api_base}/address/{address}/utxo"
            logger.info(f"🔍 获取UTXO: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"📊 Blockstream API响应: 找到 {len(data)} 个UTXO")
            
            utxos = []
            for utxo in data:
                utxos.append({
                    'txid': utxo['txid'],
                    'vout': utxo['vout'],
                    'value': utxo['value'],  # satoshis
                    'scriptPubKey': ''  # Blockstream API不直接提供脚本
                })
                logger.info(f"  - UTXO: {utxo['txid']}:{utxo['vout']} = {utxo['value']} sats")
            
            logger.info(f"🎯 最终找到 {len(utxos)} 个可用UTXO")
            return utxos
            
        except Exception as e:
            logger.error(f"获取UTXO失败: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"HTTP状态码: {e.response.status_code}")
                logger.error(f"响应内容: {e.response.text}")
                
                # 如果是404，尝试检查地址信息
                if e.response.status_code == 404:
                    logger.info("🔍 尝试获取地址基本信息...")
                    try:
                        info_url = f"{self.api_base}/address/{address}"
                        info_response = requests.get(info_url)
                        if info_response.status_code == 200:
                            info_data = info_response.json()
                            logger.info(f"📊 地址信息: {info_data}")
                        else:
                            logger.warning(f"⚠️ 地址信息也获取失败: {info_response.status_code}")
                    except Exception as info_e:
                        logger.warning(f"⚠️ 获取地址信息失败: {info_e}")
            return []
    
    async def _get_fee_rate(self):
        """获取当前推荐的交易费率 (satoshis per byte)"""
        try:
            # 使用 Blockstream API 获取费率推荐
            response = requests.get(f"{self.api_base}/fee-estimates")
            response.raise_for_status()
            data = response.json()
            
            # 获取1块确认的费率 (sat/vB)
            fee_rate = data.get('1', data.get('2', data.get('3', 20)))
            logger.info(f"💸 获取到费率: {fee_rate} sat/vB")
            return int(fee_rate)
            
        except Exception as e:
            logger.warning(f"获取费率失败，使用默认值: {e}")
            return 50 if self.network == "mainnet" else 20  # 测试网使用较低费率
    
    async def _broadcast_transaction(self, tx_hex: str):
        """广播交易到网络"""
        try:
            # 使用 Blockstream API 广播交易
            response = requests.post(
                f"{self.api_base}/tx",
                data=tx_hex,
                headers={'Content-Type': 'text/plain'}
            )
            response.raise_for_status()
            
            # Blockstream API 直接返回交易哈希
            tx_hash = response.text.strip()
            if len(tx_hash) == 64:  # 验证哈希长度
                return tx_hash
            else:
                raise Exception(f"广播响应格式错误: {tx_hash}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"广播交易失败: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"错误详情: {e.response.text}")
            raise Exception(f"交易广播失败: {e}")
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str = "BTC") -> str:
        """执行Bitcoin转账"""
        try:
            # 支持的代币检查
            supported_tokens = ["BTC", "XTN"]
            if token not in supported_tokens:
                raise ValueError(f"Bitcoin适配器不支持代币: {token}，支持的代币: {supported_tokens}")
            
            # XTN在testnet3上等同于BTC的处理方式
            display_token = token
            logger.info(f"🔄 Bitcoin转账 ({self.network}): {amount} {display_token} 到 {to_address}")
            
            # 解析私钥
            private_key_obj = CBitcoinSecret(private_key)
            from_address = self.get_address_from_private_key(private_key)
            logger.info(f"💳 发送方地址: {from_address}")
            
            # 获取UTXO
            utxos = await self._get_utxos(from_address)
            if not utxos:
                raise ValueError(f"地址 {from_address} 没有可用的UTXO")
            
            # 计算金额（转换为satoshis）
            amount_satoshis = int(float(amount) * COIN)
            logger.info(f"💰 转账金额: {amount} {display_token} ({amount_satoshis} satoshis)")
            
            # 选择UTXO
            selected_utxos = []
            total_input = 0
            for utxo in utxos:
                selected_utxos.append(utxo)
                total_input += utxo['value']
                if total_input >= amount_satoshis:
                    break
            
            if total_input < amount_satoshis:
                raise ValueError(f"余额不足: 需要 {amount_satoshis}, 可用 {total_input}")
            
            # 获取费率并估算费用
            fee_rate = await self._get_fee_rate()
            estimated_size = len(selected_utxos) * 180 + 2 * 34 + 10  # 估算交易大小
            fee = estimated_size * fee_rate
            
            # 计算找零
            change = total_input - amount_satoshis - fee
            logger.info(f"💸 交易费用: {fee} satoshis ({fee_rate} sat/byte)")
            logger.info(f"🔄 找零: {change} satoshis")
            
            if change < 0:
                raise ValueError(f"余额不足支付费用: 总额 {total_input}, 转账 {amount_satoshis}, 费用 {fee}")
            
            # 构建交易输入
            txins = []
            for utxo in selected_utxos:
                txin = CMutableTxIn(COutPoint(lx(utxo['txid']), utxo['vout']))
                txins.append(txin)
            
            # 构建交易输出
            txouts = []
            
            # 主要输出 - 转账目标
            logger.info(f"🎯 目标地址: {to_address}")
            to_script = self._create_script_pubkey_for_address(to_address)
            txout_main = CMutableTxOut(amount_satoshis, to_script)
            txouts.append(txout_main)
            
            # 找零输出（如果需要）
            dust_threshold = 546
            if change > dust_threshold:
                from_addr = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                txout_change = CMutableTxOut(change, from_addr.to_scriptPubKey())
                txouts.append(txout_change)
            elif change > 0:
                logger.warning(f"⚠️ 找零 {change} satoshis 低于粉尘阈值 {dust_threshold}，将被包含在交易费中")
            
            # 创建未签名交易
            tx = CMutableTransaction(txins, txouts)
            logger.info("✍️ 创建未签名交易")
            
            # 检测发送地址类型
            from_address_type = self._detect_address_type(from_address)
            logger.info(f"🔍 发送地址类型: {from_address_type}")
            
            # 根据地址类型处理签名
            if from_address_type == "p2wpkh":
                # SegWit P2WPKH 签名
                logger.info("✍️ 使用 P2WPKH SegWit 签名方法")
                
                # 创建 witness 列表
                from bitcoin.core import CScriptWitness, CTxInWitness, CTxWitness
                witness_list = []
                
                # 签名每个输入
                for i, (txin, utxo) in enumerate(zip(txins, selected_utxos)):
                    logger.info(f"✍️ 签名输入 {i}")
                    
                    # P2WPKH: 使用 redeemScript (P2PKH 格式)
                    import hashlib

                    from bitcoin.core.script import OP_CHECKSIG, OP_DUP, OP_EQUALVERIFY, OP_HASH160
                    from bitcoin.wallet import CBech32BitcoinAddress

                    # 创建 P2WPKH 地址对象
                    pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(private_key_obj.pub).digest()).digest()
                    from_script = CScript([OP_0, pubkey_hash])  # P2WPKH scriptPubKey
                    from_address_obj = CBech32BitcoinAddress.from_scriptPubKey(from_script)
                    
                    # 使用库提供的 SegWit 签名方法
                    sighash = SignatureHash(from_address_obj.to_redeemScript(), tx, i, SIGHASH_ALL, utxo['value'], SIGVERSION_WITNESS_V0)
                    
                    # 签名
                    signature = private_key_obj.sign(sighash) + bytes([SIGHASH_ALL])
                    
                    # SegWit: scriptSig 为空
                    txin.scriptSig = CScript()
                    
                    # 创建 witness
                    script_witness = CScriptWitness([signature, private_key_obj.pub])
                    witness_list.append(CTxInWitness(script_witness))
                
                # 设置整个交易的 witness
                tx.wit = CTxWitness(witness_list)
                
            else:
                # 传统 P2PKH 签名
                logger.info("✍️ 使用传统 P2PKH 签名方法")
                
                # 签名每个输入
                for i, (txin, utxo) in enumerate(zip(txins, selected_utxos)):
                    logger.info(f"✍️ 签名输入 {i}")
                    
                    from_addr = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                    scriptPubKey = from_addr.to_scriptPubKey()
                    
                    # 创建签名哈希
                    sighash = SignatureHash(scriptPubKey, tx, i, SIGHASH_ALL)
                    
                    # 签名
                    signature = private_key_obj.sign(sighash) + bytes([SIGHASH_ALL])
                    
                    # 创建解锁脚本
                    scriptSig = CScript([signature, private_key_obj.pub])
                    
                    # 设置输入的解锁脚本
                    txin.scriptSig = scriptSig
            
            logger.info("✍️ 交易已签名")
            
            # 序列化交易
            tx_serialized = tx.serialize()
            tx_hex = b2x(tx_serialized)
            
            # 广播交易
            tx_hash = await self._broadcast_transaction(tx_hex)
            logger.info(f"✅ {display_token}转账成功: {tx_hash}")
            logger.info(f"🔍 查看交易: {self.explorer_url}/{tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"❌ Bitcoin转账失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def diagnose_private_key(self, private_key: str) -> None:
        """诊断私钥格式和相关信息"""
        logger.info("🔍 开始私钥诊断...")
        logger.info(f"📝 私钥长度: {len(private_key)}")
        logger.info(f"📝 私钥前缀: {private_key[:3]}...")
        logger.info(f"📝 当前网络: {self.network}")
        
        expected_prefixes = self._get_wif_prefixes()
        logger.info(f"📝 期望前缀: {expected_prefixes}")
        
        if any(private_key.startswith(prefix) for prefix in expected_prefixes):
            logger.info("✅ 私钥前缀匹配")
        else:
            logger.warning("⚠️ 私钥前缀不匹配")
        
        try:
            private_key_obj = CBitcoinSecret(private_key)
            logger.info("✅ 私钥格式有效")
            logger.info(f"📝 公钥长度: {len(private_key_obj.pub)} bytes")
            logger.info(f"📝 公钥类型: {'压缩' if len(private_key_obj.pub) == 33 else '未压缩'}")
            
            # 生成不同类型地址
            for addr_type in ["p2pkh", "p2wpkh"]:
                try:
                    addr = self.get_address_from_private_key(private_key, addr_type)
                    logger.info(f"✅ {addr_type.upper()} 地址: {addr}")
                except Exception as e:
                    logger.error(f"❌ {addr_type.upper()} 地址生成失败: {e}")
                    
        except Exception as e:
            logger.error(f"❌ 私钥格式无效: {e}")
            logger.info("💡 需要转换为 Bitcoin WIF 格式")

    def _create_script_pubkey_for_address(self, address: str):
        """为不同类型的地址创建 scriptPubKey"""
        try:
            logger.info(f"🔍 处理地址: {address}")
            logger.info(f"📝 地址长度: {len(address)}")
            logger.info(f"📝 地址前缀: {address[:6] if len(address) >= 6 else address}")
            
            if address.startswith(('1', 'm', 'n')):
                # Legacy P2PKH 地址
                from bitcoin.base58 import b58decode_check
                from bitcoin.core.script import OP_CHECKSIG, OP_DUP, OP_EQUALVERIFY, OP_HASH160

                # 解码 Base58Check 地址
                addr_bytes = b58decode_check(address)
                pubkey_hash = addr_bytes[1:]  # 去掉版本字节
                
                # 创建 P2PKH scriptPubKey: OP_DUP OP_HASH160 <pubkey_hash> OP_EQUALVERIFY OP_CHECKSIG
                script = CScript([OP_DUP, OP_HASH160, pubkey_hash, OP_EQUALVERIFY, OP_CHECKSIG])
                logger.info(f"✅ 创建 P2PKH scriptPubKey for {address}")
                return script
                
            elif address.startswith(('3', '2')):
                # P2SH 地址（暂不支持）
                raise ValueError(f"P2SH 地址暂不支持: {address}")
                
            elif address.startswith(('bc1', 'tb1')):
                # Bech32 SegWit 地址
                from bitcoin.core.script import OP_0

                # 解码 bech32 地址
                hrp = "bc" if address.startswith("bc1") else "tb"
                logger.info(f"🔍 使用 HRP: {hrp}")
                decoded = self._decode_bech32(address, hrp)
                logger.info(f"🔍 Bech32 解码结果: {decoded}")
                
                if decoded is None:
                    raise ValueError(f"无效的 bech32 地址: {address}")
                
                witness_version, witness_program = decoded
                logger.info(f"🔍 Witness 版本: {witness_version}, 程序长度: {len(witness_program)}")
                
                if witness_version == 0 and len(witness_program) == 20:
                    # P2WPKH: OP_0 <20-byte-pubkey-hash>
                    script = CScript([OP_0, witness_program])
                    logger.info(f"✅ 创建 P2WPKH scriptPubKey for {address}")
                    return script
                elif witness_version == 0 and len(witness_program) == 32:
                    # P2WSH: OP_0 <32-byte-script-hash>
                    script = CScript([OP_0, witness_program])
                    logger.info(f"✅ 创建 P2WSH scriptPubKey for {address}")
                    return script
                elif witness_version == 1 and len(witness_program) == 32:
                    # P2TR (Taproot): OP_1 <32-byte-taproot-output>
                    script = CScript([1, witness_program])  # OP_1 = 1
                    logger.info(f"✅ 创建 P2TR (Taproot) scriptPubKey for {address}")
                    return script
                else:
                    raise ValueError(f"不支持的 witness 版本或程序长度: v{witness_version}, {len(witness_program)} bytes")
                    
            else:
                raise ValueError(f"不支持的地址格式: {address}")
                
        except Exception as e:
            logger.error(f"❌ 创建 scriptPubKey 失败: {e}")
            raise

    def _decode_bech32(self, address: str, hrp: str):
        """解码 Bech32 地址 (支持 Bech32 和 Bech32m)"""
        try:
            logger.info(f"🔍 开始解码地址: {address}")
            logger.info(f"🔍 使用 HRP: {hrp}")
            
            # Bech32 字符集
            CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
            
            # Bech32 生成多项式
            GENERATOR = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
            
            def bech32_polymod(values):
                chk = 1
                for value in values:
                    top = chk >> 25
                    chk = (chk & 0x1ffffff) << 5 ^ value
                    for i in range(5):
                        chk ^= GENERATOR[i] if ((top >> i) & 1) else 0
                return chk
            
            def bech32_hrp_expand(hrp):
                return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
            
            def bech32_verify_checksum(hrp, data):
                # Bech32 校验和常数是 1
                polymod_result = bech32_polymod(bech32_hrp_expand(hrp) + data)
                return polymod_result == 1
                
            def bech32m_verify_checksum(hrp, data):
                # Bech32m 校验和常数是 0x2bc830a3
                polymod_result = bech32_polymod(bech32_hrp_expand(hrp) + data)
                return polymod_result == 0x2bc830a3
            
            def convertbits(data, frombits, tobits, pad=True):
                acc = 0
                bits = 0
                ret = []
                maxv = (1 << tobits) - 1
                max_acc = (1 << (frombits + tobits - 1)) - 1
                for value in data:
                    if value < 0 or (value >> frombits):
                        return None
                    acc = ((acc << frombits) | value) & max_acc
                    bits += frombits
                    while bits >= tobits:
                        bits -= tobits
                        ret.append((acc >> bits) & maxv)
                if pad:
                    if bits:
                        ret.append((acc << (tobits - bits)) & maxv)
                elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
                    return None
                return ret
            
            # 检查地址格式
            if ((address[:len(hrp)] != hrp) or 
                (not (6 <= len(address) <= 90)) or 
                (address[len(hrp)] != '1')):
                logger.error(f"❌ 地址格式检查失败")
                return None
            
            # 分离数据部分
            data_part = address[len(hrp)+1:]
            logger.info(f"🔍 数据部分: {data_part}")
            
            # 转换字符为5位值
            data = []
            for char in data_part:
                if char not in CHARSET:
                    logger.error(f"❌ 无效字符: {char}")
                    return None
                data.append(CHARSET.index(char))
            
            logger.info(f"🔍 转换后的数据: {data[:10]}...") # 只显示前10个
            
            # 验证校验和 (先尝试 Bech32，再尝试 Bech32m)
            is_bech32 = bech32_verify_checksum(hrp, data)
            is_bech32m = bech32m_verify_checksum(hrp, data)
            
            logger.info(f"🔍 Bech32 校验: {is_bech32}")
            logger.info(f"🔍 Bech32m 校验: {is_bech32m}")
            
            if not (is_bech32 or is_bech32m):
                logger.error(f"❌ 校验和验证失败")
                return None
            
            # 提取载荷（去掉校验和）
            payload = data[:-6]
            
            if len(payload) < 1:
                logger.error(f"❌ 载荷太短")
                return None
            
            # 提取 witness 版本和程序
            witness_version = payload[0]
            if witness_version > 16:
                logger.error(f"❌ 无效的 witness 版本: {witness_version}")
                return None
            
            logger.info(f"🔍 Witness 版本: {witness_version}")
            
            # 转换 witness 程序
            witness_program = convertbits(payload[1:], 5, 8, False)
            if witness_program is None:
                logger.error(f"❌ 转换 witness 程序失败")
                return None
            
            witness_program_bytes = bytes(witness_program)
            logger.info(f"🔍 Witness 程序长度: {len(witness_program_bytes)}")
            
            # 验证程序长度
            if len(witness_program_bytes) < 2 or len(witness_program_bytes) > 40:
                logger.error(f"❌ 无效的程序长度: {len(witness_program_bytes)}")
                return None
            
            # 版本 0 只能是 20 或 32 字节
            if witness_version == 0 and len(witness_program_bytes) not in [20, 32]:
                logger.error(f"❌ 版本 0 程序长度无效: {len(witness_program_bytes)}")
                return None
                
            # 版本 1 (Taproot) 必须是 32 字节且使用 Bech32m
            if witness_version == 1:
                if len(witness_program_bytes) != 32:
                    logger.error(f"❌ 版本 1 程序长度必须是 32 字节: {len(witness_program_bytes)}")
                    return None
                if not is_bech32m:
                    logger.error(f"❌ 版本 1 必须使用 Bech32m 编码")
                    return None
            
            logger.info(f"✅ 成功解码: v{witness_version}, {len(witness_program_bytes)} bytes")
            return witness_version, witness_program_bytes
            
        except Exception as e:
            logger.error(f"❌ Bech32解码失败: {e}")
            return None

    def _detect_address_type(self, address: str) -> str:
        """检测地址类型"""
        if address.startswith(('1', 'm', 'n')):
            return "p2pkh"
        elif address.startswith(('3', '2')):
            return "p2sh"  
        elif address.startswith(('bc1q', 'tb1q')):
            return "p2wpkh"  # Native SegWit
        elif address.startswith(('bc1p', 'tb1p')):
            return "p2tr"    # Taproot
        else:
            return "unknown"


class ArbitrageExecutor:
    
    def __init__(self, network: str = "mainnet"):
        """
        初始化套利执行器
        
        Args:
            network: 网络类型，支持 "mainnet", "testnet", "testnet3", "testnet4"
        """
        if network not in ["mainnet", "testnet", "testnet3", "testnet4"]:
            raise ValueError(f"不支持的网络: {network}，请使用 'mainnet', 'testnet', 'testnet3' 或 'testnet4'")
        
        self.network = network
        self.adapters = {
            "SOL": SolanaAdapter(),
            "BTC": BitcoinAdapter(network=network),
            "XTN": BitcoinAdapter(network=network),
        }
    
    def get_adapter(self, chain: str) -> ChainAdapter:
        if chain not in self.adapters:
            raise ValueError(f"不支持的区块链: {chain}")
        return self.adapters[chain]
    
    async def execute_arbitrage(self, fluxlayer_exchange, trading_pair):
        try:
            pair_meta = fluxlayer_exchange.metadata.trading_pairs[trading_pair]
            i_amount = fluxlayer_exchange.metadata.source_amount
            o_amount = fluxlayer_exchange.metadata.target_amount
            is_buy = fluxlayer_exchange.metadata.is_buy
            # 根据 is_buy 决定交易方向
            if is_buy:
                # 买入操作：从 source 到 target
                src_chain = pair_meta.source_chain
                target_chain = pair_meta.target_chain
                i_token = pair_meta.source_token
                o_token = pair_meta.target_token
                direction = "BUY"
            else:
                # 卖出操作：从 target 到 source
                src_chain = pair_meta.target_chain
                target_chain = pair_meta.source_chain
                i_token = pair_meta.target_token
                o_token = pair_meta.source_token
                direction = "SELL"

            # 从环境变量获取Solana私钥
            private_key = get_private_key_from_env(src_chain)

            if not private_key:
                logger.error("❌ 无法获取SOL私钥，套利执行中止")
                return

            logger.info(f"🚀 开始套利执行 {trading_pair} ({direction})")
            logger.info(f"📊 源链: {src_chain} -> 目标链: {target_chain}")
            logger.info(f"💱 转换: {i_amount} {i_token} -> {o_amount} {o_token}")

            # 获取源链适配器
            src_adapter = self.get_adapter(src_chain)
            
            # 获取源地址
            src_addr = src_adapter.get_address_from_private_key(private_key)
            logger.info(f"👛 源地址 ({src_chain}): {src_addr}")

            # 初始化 MPC 客户端
            mpc_client = MPCClient()

            # 检查或创建 MPC 钱包
            logger.info("🔍 检查MPC钱包存在性...")
            if not mpc_client.check_mpc_exists(src_addr):
                logger.info(f"🆕 创建MPC钱包 for {src_addr}")
                wallet_id = mpc_client.create_mpc_wallet(src_addr)
                logger.info(f"✅ MPC钱包已创建，ID: {wallet_id}")
            else:
                wallet_id = mpc_client.get_mpc_wallet_id(src_addr)
                logger.info(f"✅ 找到现有MPC钱包，ID: {wallet_id}")

            # 获取 MPC 存款地址
            logger.info(f"🔍 查找MPC存款地址 for {src_chain}...")
            mpc_deposit_addr = mpc_client.find_mpc_addr(src_addr, src_chain)
            logger.info(f"🏦 MPC {src_chain} 存款地址: {mpc_deposit_addr}")
            
            # 验证 MPC 地址格式
            try:
                if src_chain == "SOL":
                    PublicKey(mpc_deposit_addr)
                    logger.info("✅ MPC地址格式有效")
            except Exception as e:
                logger.error(f"❌ 无效的MPC地址格式: {e}")
                return

            # 估算交易费用
            logger.info("💰 估算交易费用...")
            try:
                token_id = i_token
                if src_chain != i_token:
                    token_id = src_chain + "_" + i_token
                fee = mpc_client.estimate_tx_fee(wallet_id,token_id, i_amount, mpc_deposit_addr)
                total_deposit = float(i_amount) + float(fee)
                logger.info(f"💸 估算费用: {fee} {i_token}")
                logger.info(f"💰 总存款需求: {total_deposit} {i_token} ({i_amount} + {fee} 费用)")
            except Exception as e:
                logger.warning(f"⚠️ 费用估算失败: {e}, 使用默认费用")
                fee = 0.001 if i_token == "SOL" else 0
                total_deposit = float(i_amount) + fee

            # 执行代币转账到 MPC 地址
            logger.info(f"🔄 开始转账到MPC地址...")
            tx_hash = await src_adapter.transfer_token(
                private_key=private_key, 
                to_address=mpc_deposit_addr,
                amount=str(total_deposit),
                token=i_token
            )
            
            if not tx_hash:
                logger.error(f"❌ 转账到MPC地址失败")
                return
                
            logger.info(f"✅ 存款交易完成: {tx_hash}")

            # 签名目标订单
            logger.info("✍️ 签名maker订单...")
            try:
                message = f"{target_chain}{o_token}{o_amount}".encode("utf-8")
                sig = src_adapter.sign_message(private_key, message)
                logger.info(f"✅ 订单签名生成: {sig[:16]}...")
            except Exception as e:
                logger.error(f"❌ 订单签名失败: {e}")
                return

            # 创建maker订单
            logger.info("🔨 创建maker订单...")
            logger.info(f"📋 订单参数:")
            logger.info(f"   tx_hash: {tx_hash}")
            logger.info(f"   src_chain: {src_chain}")
            logger.info(f"   target_chain: {target_chain}")
            logger.info(f"   o_token: {o_token}")
            logger.info(f"   o_amount: {o_amount}")
            logger.info(f"   slippage: 0.01")
            logger.info(f"   sig: {sig[:32]}...")
            
            try:
                result = mpc_client.create_maker_order(
                    tx_hash=tx_hash,
                    src_chain=src_chain,
                    target_chain=target_chain,
                    o_token=o_token,
                    o_amount=str(o_amount),  # 确保是字符串
                    slippage="0.01",
                    sig=sig
                )
                logger.info(f"✅ Maker订单创建成功: {result}")
                
            except Exception as e:
                logger.error(f"❌ Maker订单创建失败: {e}")
                # 如果是requests异常，打印响应内容
                if hasattr(e, 'response'):
                    try:
                        error_detail = e.response.json()
                        logger.error(f"❌ 错误详情: {error_detail}")
                    except:
                        logger.error(f"❌ 响应内容: {e.response.text}")
                return

        except Exception as e:
            logger.error(f"💥 套利执行失败: {str(e)}")
            import traceback
            traceback.print_exc()

async def execute_arbitrage_with_fluxlayer(trading_pair: str = "XTN-USDC"):
    fluxlayer_exchange = MockFluxLayerExchange()
    # 创建套利执行器并执行
    executor = ArbitrageExecutor(network="testnet4")
    await executor.execute_arbitrage(fluxlayer_exchange, trading_pair)

if __name__ == "__main__":
    async def main():
       await execute_arbitrage_with_fluxlayer("XTN-USDC")
    
    asyncio.run(main())