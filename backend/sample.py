import motor.motor_asyncio
from fastapi import FastAPI


app = FastAPI()

# uri = "mongodb+srv://apurvabackend:1234@task3.lnfrkc5.mongodb.net/?retryWrites=true&w=majority" 
uri = "mongodb+srv://apurvabackend:1234@task3.lnfrkc5.mongodb.net/?retryWrites=true&w=majority"


# Create a new client and connect to the server 
client = motor.motor_asyncio.AsyncIOMotorClient(uri) 

# Send a ping to confirm a successful connection 
try:     
    client.admin.command('ping')     
    print("Pinged your deployment. You have successfully connected to MongoDB!") 
except Exception as e:     
    print(e) 