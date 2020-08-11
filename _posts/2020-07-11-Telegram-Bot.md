---
comments: true
title: Telegram Bot
tags: [Tips N Tricks, APIs]
style: fill
color: secondary
description: Instead of printing something, send it to telegram. This is very useful if you have remote machines for training.
---

```python
import requests

def send_message(msg, chat_id, bot_token):
    """
    params:
    -------
    msg: message you want to receive
    chat_id: CHAT_ID
    bot_token: API_KEY of your bot
    """

    url  = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {'chat_id': str(chat_id), 'text': f'{msg}'}
    requests.post(url, data)
```

This is all the code you need. I know its not a lot of code but same is the case with many useful things. Useful things are simple by design. I think this is very important as the use-cases are infinite. Instead of using `print()` you can use `send_message()` from above. I called it after each epoch to get the `val_loss` and `train_loss` while training models on colab. Heres how it looks:

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6698874678859112448" height="656" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

Generally speaking, you can also use it to notify you ABOUT ANYTHING.

### Finding *chat_id* and *bot_token* . . . 

As you might have already realized the above won't work without *chat_id* and *bot_token*. So, how do you find them?

First, we will have to create a bot. You need to use “TheBotfather” in the Telegram app. You can find it [here](https://telegram.me/botfather) or by searching for “botfather” in the Telegram app.

In the chat with The Botfather, enter “/newbot”. It will ask you for a name, then a username for your bot. Once you provide both, The Botfather will provide you with a link to your bot and an API token.

![image](assets/ankur.jpg)

Follow the link to your bot by clicking the link that looks like `t.me/{yourBotUsername}`. This is where you will receive messages.

We need to send it a message to enable the bot to message us back. So, go ahead and send a message to your bot. We will read this message via the Telegram API to get the `chat_id`.

Visit `https://api.telegram.org/bot{API_KEY}/getUpdates` via browser or curl (unix command) after sending the message. 

It should then return a JSON object that looks something like this:

```
{"ok":true,"result":[{
    "update_id":36317827,
    "message":{"message_id":407,
               "from":{"id":{CHAT_ID},"is_bot":false,"first_name":"Ankur","last_name":"Singh","language_code":"en"},
               "chat":{"id":{CHAT_ID},"first_name":"Ankur","last_name":"Singh","type":"private"},
               "date":1597155237,
               "text":"Hi bot"
               }
    }]
}
```

You would get a empty *result* if you forget to send message earlier.

```
{"ok":true,"result":[]}
```

Take the {CHAT_ID} value, its a 9 digit number.

Now, you have your `BOT_TOKEN` (i.e. API_KEY) and `CHAT_ID`.

### Word of caution
Never hard-code or share your `BOT_TOKEN` and `CHAT_ID`. You can either set them a environment variables or pass them from the command line (as arguments).

### Final words
Once you have your `BOT_TOKEN` and `CHAT_ID`, you can simply use `send_message()`. 
If you found it useful and happen to use it, then let everyone know by sharing your use-cases in the comment section below. Looking forward to hear from you.
