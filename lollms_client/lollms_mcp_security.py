# File name: lollms_mcp_security.py
# Author: parisneo

"""
This script defines a custom token verifier for MCP (Model Context Protocol) using an introspection endpoint.
The verifier queries the authorization server to check if a given token is valid. It is agnostic about how tokens are created.

Key components:
- MyTokenInfo class: Extends AccessToken and includes additional fields like user_id and username.
- IntrospectionTokenVerifier class: Implements the logic to verify tokens by making HTTP requests to an introspection endpoint.
- token_info_context: A context variable to store token information for easy access.

The script also includes an example of how to use these components within a FastMCP instance, setting up authentication and authorization settings.

Dependencies:
- mcp.server.auth.provider
- httpx
- os
- contextvars

Environment Variables:
- AUTHORIZATION_SERVER_URL: The URL of the authorization server. Default is 'http://localhost:9642'.
"""

from mcp.server.auth.provider import AccessToken, TokenVerifier
import httpx
import os
from contextvars import ContextVar

AUTHORIZATION_SERVER_URL = os.environ.get("AUTHORIZATION_SERVER_URL", "http://localhost:9642")

class MyTokenInfo(AccessToken):
    user_id: int | None = None
    username: str | None = None

token_info_context: ContextVar[MyTokenInfo | None] = ContextVar("token_info_context", default=None)

# This is our set of valid API keys. In a real app, you'd check a database.
class IntrospectionTokenVerifier(TokenVerifier):
    """
    This verifier asks the authorization server if a token is valid.
    It is completely agnostic about how tokens are created.
    """
    async def verify_token(self, token: str) -> AccessToken:
        # Make a secure HTTP call to your /introspect endpoint
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{AUTHORIZATION_SERVER_URL}/api/auth/introspect",
                    data={"token": token}
                )
                response.raise_for_status()  # Raise an exception for 4xx/5xx errors
            except httpx.RequestError as e:
                print(f"ERROR: Could not connect to introspection endpoint: {e}")
                return AccessToken(active=False, token="", client_id="", scopes=[])

        # Create a TokenInfo object directly from the JSON response
        token_info_dict = response.json()
        token_info_dict["token"] = token
        token_info_dict["client_id"] = str(token_info_dict.get("user_id"))
        token_info_dict["scopes"] = []
        token_info = MyTokenInfo(**token_info_dict)
        token_info_context.set(token_info)
        return MyTokenInfo(**token_info_dict)

# To recover the user information, just use token_info = token_info_context.get()
# Example use
# resource_server_url=f"http://localhost:{port}"
# mcp = FastMCP(
#     name="MyMCPServer",
#     host=host,
#     port=port,
#     log_level=log_level,
#     # 1. This tells MCP to use our class for authentication.
#     token_verifier=IntrospectionTokenVerifier(),
#     # 2. This tells MCP to protect all tools by default.
#     auth=AuthSettings(
#         # The URL of the server that issues tokens
#         issuer_url=AUTHORIZATION_SERVER_URL,
#         # The URL of the MCP server itself
#         resource_server_url=resource_server_url,  # The port of the MCP server
#         required_scopes=[]  # Requires valid authentication
#     )
# )
