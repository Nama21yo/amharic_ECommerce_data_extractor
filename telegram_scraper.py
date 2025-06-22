
import asyncio
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import pandas as pd
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

# Telegram API credentials
api_id = os.getenv('TELEGRAM_API_ID')
api_hash = os.getenv('TELEGRAM_API_HASH')
phone = os.getenv('TELEGRAM_PHONE')

# List of Telegram channels
channels = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'Shewabrand', 'helloomarketethiopia', 'modernshoppingcenter', 'qnashcom'
]

async def scrape_telegram():
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start(phone=phone)
    
    data = []
    for channel in channels:
        try:
            entity = await client.get_entity(f'@{channel}')
            async for message in client.iter_messages(entity, limit=100):  # Limit to 100 messages per channel
                media_type = None
                media_path = None
                if message.media:
                    if isinstance(message.media, MessageMediaPhoto):
                        media_type = 'photo'
                        media_path = f'media/photos/{message.id}.jpg'
                        await message.download_media(file=media_path)
                    elif isinstance(message.media, MessageMediaDocument):
                        media_type = 'document'
                        media_path = f'media/documents/{message.id}'
                        await message.download_media(file=media_path)
                
                data.append({
                    'channel': channel,
                    'sender': message.sender_id,
                    'timestamp': message.date,
                    'message': message.text,
                    'media_type': media_type,
                    'media_path': media_path
                })
        except Exception as e:
            print(f"Error scraping {channel}: {e}")
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('data/raw_telegram_data.csv', index=False, encoding='utf-8')
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(scrape_telegram())
