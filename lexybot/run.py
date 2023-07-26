import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import discord
import httpx
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores import MilvusVectorStore
from loguru import logger
from pydantic import BaseModel

load_dotenv()

logger.level("INFO")


intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)

queue = asyncio.Queue()

openai.api_key = os.environ["OPENAI_API_KEY"]


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


@bot.event
async def on_ready():
    logger.info(f"{bot.user.name} has connected to Discord!")
    asyncio.get_running_loop().create_task(background_task())


@bot.event
async def on_message(message: discord.Message):
    try:
        if message.author == bot.user:
            return

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
        logger.exception(f"An error occurred: {e}")


@bot.slash_command(guild_ids=[os.environ["DISCORD_GUILD"]])
async def lexy(ctx):
    # Health check
    await ctx.respond("Hello!")


def query(message: discord.Message):
    vector_store = MilvusVectorStore(
        host=os.environ["MILVUS_HOST"],
        port=os.environ["MILVUS_PORT"],
        user="db_admin",
        password=os.environ["MILVUS_PASSWORD"],
        use_secure=True,
        collection_name="lex",
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever()
    nodes = retriever.retrieve(message.clean_content)
    processor = SimilarityPostprocessor(similarity_cutoff=0.95)
    filtered_nodes = processor.postprocess_nodes(nodes)

    if len(filtered_nodes) != 0:
        response = ""
        for node in filtered_nodes:
            response += node.node.get_content() + "\n"
        logger.info(f"Response:", response)
        return response

    return None


def chat(
    messages: List[Message],
    ctx: Optional[str] = None,
) -> str:
    try:
        client = httpx.Client(timeout=180.0)
        reqUrl = os.environ["LLM_URL"]
        headersList = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"messages": messages, "ctx": ctx})

        logger.info(f"Request URL: {reqUrl}")
        logger.info(f"Request Payload: {payload}")

        data = client.post(reqUrl, data=payload, headers=headersList)
        return data.text
    except Exception as e:
        logger.exception(e)
        return "Sorry, I couldn't process your request at the moment."


async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

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

        try:
            ctx = query(msg)
            logger.info(f"Context: {ctx}")
        except Exception as e:
            logger.exception(e)
            ctx = None

        response = await loop.run_in_executor(
            executor, chat, [dict(m) for m in messages], ctx
        )

        logger.info(f"Response: {response}")
        await msg.reply(response)


bot.run(os.environ["DISCORD_TOKEN"])
