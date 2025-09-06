from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)

async def check_connection():
    # List database names
    dbs = await client.list_databases()
    print("Databases:", dbs)

# Run in Python shell or main
import asyncio
asyncio.run(check_connection())
