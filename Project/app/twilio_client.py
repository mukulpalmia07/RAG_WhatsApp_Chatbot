# app/twilio_client.py
import os
import logging
from twilio.rest import Client

logger = logging.getLogger(__name__)

''' Creating a class that is a wrapper around Twilio's client. '''
class TwilioClient:
    def __init__(self, account_sid: str, auth_token: str, from_whatsapp: str):
        self.client = Client(account_sid, auth_token)
        self.from_whatsapp = from_whatsapp

    def send_whatsapp(self, to: str, body: str) -> dict:
        try:
            msg = self.client.messages.create(
                body = body,
                from_= self.from_whatsapp,
                to = to
            )
            return {"sid": msg.sid, "status": msg.status}
        except Exception as e:
            logger.exception("Failed to send Twilio message")
            raise
