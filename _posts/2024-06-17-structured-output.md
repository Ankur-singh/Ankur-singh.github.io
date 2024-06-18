---
comments: true
title: Generating Structured Output with LLMs (Part 1)
tags: [LLMs, OpenAI]
style: border
color: primary
description: LLMs are great at generating text, but how do you get them to generate structured output? 
---

Large Language Models (LLMs) excel at generating human-like text, but what if you need structured output like JSON, XML, HTML, or Markdown? Structured text is essential because computers can efficiently parse and utilize it. Fortunately, LLMs can generate structured output out-of-the-box, thanks to the vast amount of structured data in their training sets.

## Prompting for Structured Output
Here's how I used the gpt-3.5-turbo model with a tailored system prompt to produce structured output:

```python
SYSTEM_PROMPT = "You are a Programmer. Whenever asked a question, return the answer in the requested programming language or markup language. Your output will be directly used as input by other downstream computer programs. So no string delimiters wrapping it, no yapping, no markdown, no fenced code blocks."
```

**Example Generations**

```python
>> print(generate_text("Hello in html with h1 tag"))
# Output: <h1>Hello</h1>

>> print(generate_text("Hello in markdown with h1 tag"))
# Output: # Hello

>> print(generate_text("Hello in json"))
# Output: {"message": "Hello"}

>> print(generate_text("print Hello in SQL"))
# Output: SELECT 'Hello';

>> print(generate_text("write code to print Hello in Python"))
# Output: print("Hello")
```

## Instructions Aren’t Enough!
Even though LLMs are great at following instructions, they sometimes generate incorrect tokens. While natural languages can handle small mistakes, structured languages can't. One wrong comma can break your JSON object. This is where Finite State Machines (FSM) and Grammars come into play.

For decades, we've used these concepts in computer science. Your IDE or compiler uses them to catch syntax errors. We can leverage the same principles to ensure the LLM-generated output adheres to the correct grammar.

Here are the important notes about [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) from OpenAI

>    - When using JSON mode, always instruct the model to produce JSON via some message in the conversation, for example via your system message. If you don't include an explicit instruction to generate JSON, the model may generate an unending stream of whitespace and the request may run continually until it reaches the token limit. To help ensure you don't forget, the API will throw an error if the string `"JSON"` does not appear somewhere in the context.
>    - The JSON in the message the model returns may be partial (i.e. cut off) if `finish_reason` is length, which indicates the generation exceeded `max_tokens` or the conversation exceeded the token limit. To guard against this, check `finish_reason` before parsing the response.
>    - JSON mode will not guarantee the output matches any specific schema, only that it is valid and parses without errors.

Do the first two points make sense now? If we don't instruct the model, it will generate human-like text, which will often be rejected because it doesn't conform to JSON grammar.

**Note:** JSON is a widely used format, and all major LLM vendors and runtimes support it. While other formats are not supported out of the box, you can use packages like [outline](https://github.com/outlines-dev/outlines) or [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) to enforce grammar.


### Specifying Schema

Now, let's talk about the third point in the OpenAI JSON Mode notes. In most cases, you want your JSON object to have a certain schema. LLM + Grammar can reliably generate a valid JSON string but can't enforce the schema for you. The solution to this is twofold:
1. **Schema Definition:** Make sure you provide the required schema definition to the LLM during prompting. The model will try its best to stick to the provided schema.
2. **Schema Validation:** Next, you need a way to validate the JSON object against the schema.


[Pydantic](https://docs.pydantic.dev/latest/) is the most popular package in Python for schema validation. It can help you with both 1 and 2. Once you define your Pydantic model, you can call the `model_json_schema()` method to get the JSON schema of the model. Here is the code:


```python
import json
from pydantic import BaseModel

# Model definition
class Person(BaseModel):
    name: str
    age: int
    occupation: str

SYSTEM_PROMPT = """You will be given a name of a celebrity. Please return the response in the following JSON format:

{schema}

Remember, the response will be fed directly to the next downstream program. So no string delimiters wrapping it, no yapping, no markdown, no fenced code blocks.
"""

# Injecting schema definition
SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    schema=json.dumps(Person.model_json_schema(), indent=2)
)

# Calling LLM
response = generate_text("write code to print Hello in Python")

# Validating `response` string against Model Schema
response_json = json.loads(response)
person = Person(**response_json)
print(person)
```

As you can see, this approach, though effective, involves repetitive steps and is prone to human error. Hence, I personally like to use `instructor` when working with LLM to generate JSON output. The `instructor` package streamlines this process. It patches client packages of major LLM vendors to automatically handle schema injection and validation. It's a very light wrapper and does the following:
1. Updates the system prompt to produce JSON output and also adds the JSON schema for your Pydantic model.
2. Turns on JSON mode.
3. Validates the JSON output to make sure it follows the specified JSON schema.

This drastically reduces boilerplate code and minimizes the risk of errors. You just have to patch your `client` object and pass `response_model` during inference. Here is the updated code after introducing `instructor`:


```python
import instructor

# Patching
client = instructor.from_openai(openai_client)

SYSTEM_PROMPT = """You will be given a name of a celebrity, return me the details."""

response = instructor_client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,  # Extra argument to specify the JSON Schema
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
)

type(response)  # Output: Person
```

> If it helps, here is the code snippet for [System prompt](https://github.com/jxnl/instructor/blob/b5d0fd428119724c4fcbf5d27fecc72b97353c73/instructor/process_response.py#L242-L252) and the code snippet for [ingestion logic](https://github.com/jxnl/instructor/blob/b5d0fd428119724c4fcbf5d27fecc72b97353c73/instructor/process_response.py#L274-L286) to see how `instructor` injects the JSON schema into the system prompt.


## Summary
To summarize, here are the steps to ensure LLMs generate valid structured output:

- **Explicit Instruction:** Instruct your LLM to generate structured output.
- **Grammar Enforcement:** Use grammar to validate the generated text. JSON mode is widely supported and robust, but other formats may need tools like [outline](https://github.com/outlines-dev/outlines) or [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer).
- **Schema Specification:** Provide the schema during prompting and validate the output against it. Tools like Pydantic and instructor simplify this process.

### Resources
- JSON mode in [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#request-json-mode) and [vLLM](https://docs.vllm.ai/en/v0.5.0/serving/openai_compatible_server.html) (search for `guided-decoding-backend` option).
- Have not tired it, but looks interesting, [Super-json-mode](https://github.com/varunshenoy/super-json-mode).
- Read [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) by the authors of the `outline` package.
- Grammar support in [Llama.cpp](https://til.simonwillison.net/llms/llama-cpp-python-grammars).

Now, go forth and generate structured outputs like a pro! Happy coding!