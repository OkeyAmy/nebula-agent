import os
from nebula import Nebula
from typing import Optional

class NebulaClient:
    def __init__(self):
        self.client = Nebula(
            base_url=os.getenv("NEBULA_BASE_URL", "https://nebula-api.thirdweb.com"),
            secret_key=os.getenv("THIRDWEB_SECRET_KEY")
        )

    def pay_invoice(
        self,
        user_id: str,
        amount_eth: float,
        to_address: str,
        invoice_id: str,
        stream: bool = False
    ) -> Optional[str]:
        """
        Sends a payment instruction to Nebula and returns the transaction hash.
        If `stream=True`, yields incremental response chunks.
        """
        message = (
            f"Pay {amount_eth} ETH to {to_address} for Invoice #{invoice_id}"
        )
        # Non-streaming chat+execute call
        response = self.client.chat(
            message=message,
            user_id=user_id,
            stream=stream,
            execute=True
        )
        # If streaming, return generator of chunks
        if stream:
            for chunk in response:
                yield chunk
            return None

        # response.message is the LLM’s text; response.action holds the TX data
        tx_hash = getattr(response.action, "tx_hash", None)
        return tx_hash
