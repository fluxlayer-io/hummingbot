import aiohttp
import os
from hummingbot.core.web_assistant.connections.data_types import RESTRequest, RESTResponse


class RESTConnection:
    def __init__(self, aiohttp_client_session: aiohttp.ClientSession):
        self._client_session = aiohttp_client_session

    async def call(self, request: RESTRequest) -> RESTResponse:
        proxy = os.environ.get("HTTP_PROXY")
        request_kwargs = {
            "method": request.method.value,
            "url": request.url,
            "params": request.params,
            "data": request.data,
            "headers": request.headers,
        }
        if proxy:
            request_kwargs["proxy"] = proxy

        aiohttp_resp = await self._client_session.request(**request_kwargs)
        resp = await self._build_resp(aiohttp_resp)
        return resp

    @staticmethod
    async def _build_resp(aiohttp_resp: aiohttp.ClientResponse) -> RESTResponse:
        resp = RESTResponse(aiohttp_resp)
        return resp
