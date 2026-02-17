# ================================
# Autobot Telegram Bot Example
# ================================

# WHAT:
# This script creates a simple Telegram bot using the Autobot framework.
# The bot continuously polls Telegram for new messages and replies to them.

# WHY:
# Telegram bots are automated accounts that can respond to user messages.
# By running this script, you can interact with your bot in the Telegram app.
# Autobot provides a Pythonic way to connect to Telegram's API asynchronously.

# HOW:
# 1. Install Autobot: pip install git+https://github.com/osckampala/autobot.git
# 2. Create a bot with @BotFather in Telegram and get your token.
# 3. Replace TOKEN below with your bot’s token.
# 4. Run this script: python bot.py
# 5. Open Telegram, search for your bot, and send it messages.

import asyncio
from autobot.telegram.context import Context

# Replace with your actual token from BotFather
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

async def run_bot():
    cxt = Context(TOKEN)
    print("Bot is running... Press Ctrl+C to stop.")

    # Continuous polling loop
    while True:
        updates = await cxt.get_updates()
        for update in updates:
            if (m := update.message):
                chat_id = m.chat.id
                text = m.text
                print(f"Received: {text}")

                # Command handling
                if text == "/start":
                    await cxt.send_message(chat_id=chat_id, text="Welcome! I am your Autobot 🤖")
                elif text.lower() == "hello":
                    await cxt.send_message(chat_id=chat_id, text="Hi there 👋")
                else:
                    await cxt.send_message(chat_id=chat_id, text=f"You said: {text}")

        # Sleep briefly to avoid hitting Telegram too fast
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run_bot())
