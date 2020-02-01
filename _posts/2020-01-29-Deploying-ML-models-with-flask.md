---
comments: true
title: Deploying ML models with flask
tags: [Machine Learning, flask, web development]
style: fill
color: warning
description: Learn to deploy you machine learning model using flask.
---

Python is an amazing language to learn. It started as scripting language but now it is used for everything ranging from Web Development to Data Science and everything in-between. In this blog post we will learn to deploy a machine learning model using flask.

I have created a git [repo](#) with all the code that I will be using in this blog.

Lets get started, there are 3 primary things that we will be learning today.
- Flask basics
    - urls
    - templates
- Deploying Apps using flask
- Deploying ML models using flask

#### work in progress ...

### Flask basics

Flask is a micro web-framework in python. Its very easy to develop a basics website and get it running using flask and python. Lets dive into it

#### Hello World

```python
from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def index():
    return '<h1>hello world</h1>'

if __name__ == '__main__':
    app.run()
```

This is all the code you need to build your first website with flask. You start by creating a flask object (`app = Flask(__name__)`), this variable will act as a starting point for your web-app. Next, we will use a decorator (`@app.route('/hello')`). This decorator tells the web-app that when someone types `<hostname:port>/hello` in the URL bar of the browser, you must execute `index()` function. Basically, we are mapping routes/URLs to functions i.e. if this URL is visited, then this function has to be executed. Notice, we are using the `app` object in the decorator name. I told you, its going to be the starting point for your web-app.

#### URLs 

- **Simple route**

```python 
@app.route('/')
def index():
    return '<h1>This is the INDEX Page.</h1>'
```


- **Dynamic route**

```python
@app.route(<string:name>)
def profile(name):
    return f"<h1>This is the {name}'s profile.</h1>"
```

#### Templates

- **Rendering HTML files**

```python
from flask import Flask, render_template

@app.route('/')
def index():
    return render_template('index.html')
```

- **Passing values to the front-end**

```python
@app.route('/heading')
def heading():
    heading = 'Heading from python'
    return render_template('index2.html', heading = heading)
```

- **Extending base template**

```python
@app.route('/extend')
def jinja_extend():
    return render_template('extend.html')
```

- **Lists in jinja**

```python
@app.route('/list')
def jinja_list():
    names = ['Rahul', 'Gandhi', 'Virat', 'Conor']
    return render_template('list.html', names = names)
```

- **Conditions in jinja**

```python
@app.route('/condition')
def jinja_conditions():
    day = int(np.random.rand(1) * 10) % 2
    return render_template('condition.html', text = day)
```

#### Building an app
