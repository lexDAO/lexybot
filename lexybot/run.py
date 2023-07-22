import os
import discord
from discord.ext import commands

from dotenv import load_dotenv

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

from typing import List, Optional
from pydantic import BaseModel
import httpx
import json

load_dotenv()


intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


def chat(messages: List[Message]) -> str:
    try:
        client = httpx.Client(timeout=180.0)
        reqUrl = "https://nerderlyne--llama2-chat.modal.run"
        headersList = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"messages": messages})
        data = client.post(reqUrl, data=payload, headers=headersList)
        return data.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request at the moment."


@bot.event
async def on_ready():
    print(f"{bot.user.name} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    print(
        f"{message.channel}: {message.author}: {message.author.name}: {message.content}"
    )
    try:
        if message.author == bot.user:
            return

        if (
            bot.user.mentioned_in(message)
            or message.reference
            and message.reference.resolved.author == bot.user
        ):
            messages = [{"role": "user", "content": message.content}]
            response = chat(messages)
            print(response)
            await message.channel.send(response)
    except Exception as e:
        print(f"An error occurred: {e}")


@bot.slash_command(guild_ids=[os.environ["DISCORD_GUILD"]])
async def lexy(ctx):
    await ctx.respond("Hello!")


bot.run(os.environ["DISCORD_TOKEN"])
