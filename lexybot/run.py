import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import discord
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)

queue = asyncio.Queue()


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


@bot.event
async def on_ready():
    print(f"{bot.user.name} has connected to Discord!")
    asyncio.get_running_loop().create_task(background_task())


@bot.event
async def on_message(message: discord.Message):
    print(
        f"{message.channel}: {message.author}: {message.author.name}: {message.content}"
    )
    try:
        if message.author == bot.user:
            return
        print("Bot mentioned: ", bot.user.mentioned_in(message))
        if bot.user.mentioned_in(message):  # mentions bot
            if isinstance(message.channel, discord.channel.DMChannel) or (
                bot.user and bot.user.mentioned_in(message)
            ):
                if message.reference:
                    pastMessage = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                else:
                    pastMessage = None
                await queue.put((message, pastMessage))
    except Exception as e:
        print(f"An error occurred: {e}")


@bot.slash_command(guild_ids=[os.environ["DISCORD_GUILD"]])
async def lexy(ctx):
    await ctx.respond("Hello!")


def chat(messages: List[Message]) -> str:
    try:
        print("Chatting with LLM...")
        client = httpx.Client(timeout=180.0)
        reqUrl = os.environ["LLM_URL"]
        headersList = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"messages": messages})
        print(f"Request URL: {reqUrl}")
        print(f"Request Payload: {payload}")
        print(f"Request Headers: {headersList}")
        data = client.post(reqUrl, data=payload, headers=headersList)
        return data.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request at the moment."


async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    print("Task Started. Waiting for inputs.")
    while True:
        msg_pair: tuple[discord.Message, discord.Message] = await queue.get()
        msg, past = msg_pair

        messages = []
        if past:
            messages.append(
                Message(
                    role=past.author.name == bot.user.name and "assistant" or "user",
                    content=past.clean_content,
                    name=past.author.name,
                )
            )
        messages.append(
            Message(
                role=msg.author.name == bot.user.name and "assistant" or "user",
                content=msg.clean_content,
                name=msg.author.name,
            )
        )

        response = await loop.run_in_executor(
            executor, chat, [dict(m) for m in messages]
        )
        print(f"Response: {response}")
        await msg.reply(response)


bot.run(os.environ["DISCORD_TOKEN"])
