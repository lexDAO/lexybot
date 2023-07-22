import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

bot = commands.Bot(command_prefix="!")


@bot.event
async def on_ready():
    print(f"{bot.user.name} has connected to Discord!")


@bot.command(name="ping")
async def ping(ctx):
    await ctx.send("Pong!")


bot.run(os.environ["DISCORD_TOKEN"])
