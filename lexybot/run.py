import json
import os
from typing import List, Optional

import discord
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

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
        print("Bot mentioned: ", bot.user.mentioned_in(message))
        if bot.user.mentioned_in(message):  # mentions bot
            previous_messages = await message.channel.history(limit=3).flatten()
            messages = []
            for previous_message in reversed(previous_messages):
                role = "user"
                if message.author == bot.user:
                    role = "assistant"

                messages.append(
                    Message(
                        role=role,
                        content=previous_message.clean_content,
                        name=previous_message.author.name,
                    )
                )

            messages.append(
                Message(
                    role="user",
                    content=message.clean_content,
                    name=message.author.name,
                )
            )  # replace user id with user name

            print(messages)
            response = chat([dict(m) for m in messages])
            print(response)
            await message.reply(response)
    except Exception as e:
        print(f"An error occurred: {e}")


@bot.slash_command(guild_ids=[os.environ["DISCORD_GUILD"]])
async def lexy(ctx):
    await ctx.respond("Hello!")


bot.run(os.environ["DISCORD_TOKEN"])
