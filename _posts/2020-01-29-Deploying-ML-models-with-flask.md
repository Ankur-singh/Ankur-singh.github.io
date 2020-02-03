---
comments: true
title: Deploying ML models with flask
tags: [Machine Learning, flask, web development]
style: fill
color: primary
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

### Flask basics

Flask is a micro web-framework in python. Its very easy to develop a basics website and get it running using flask and python. Lets dive into it

**Disclaimer:** The blog is **not** intended to get you started with flask and ML model deployment. It is meant for giving you an overview of how to deploy application using flask and as a support material for the meetup. So, I will highly recommend everyone going through the blog to check out the recommended files to have a look at the complete code.

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

This is all the code you need to build your first website with flask. You start by creating a flask object `app = Flask(__name__)`, this variable will act as a starting point for your web-app. Next, we will use a decorator `@app.route('/hello')`. This decorator tells the web-app that when someone types `<hostname:port>/hello` in the URL bar of the browser, you must execute `index()` function. Basically, we are mapping routes/URLs to functions i.e. if this URL is visited, then this function has to be executed. Notice, we are using the `app` object in the decorator name. I told you, its going to be the starting point for your web-app.

#### URLs 

##### - **Simple route**
We already saw a simple route in our first example. These types of routes simply infos the web-app about the routes-functions mapping.

```python 
@app.route('/')
def index():
    return '<h1>This is the INDEX Page.</h1>'
```

##### - **Dynamic route**
Instead of having a fixed URL for every person, we specify a template for the URL. If the entered URL matches the template, then the function is executed. The matched values is pass as an argument to the associated function. Flask has support for many other variable types like **int**, **float** and **path**.

```python
@app.route(<string:name>)
def profile(name):
    return f"<h1>This is the {name}'s profile.</h1>"
```

**Recommended files:** [1_urls.py](https://github.com/Ankur-singh/flask_demo/blob/master/1_urls.py)

#### Templates

##### - **Rendering HTML files**
Flask uses [jinja template engine](https://www.palletsprojects.com/p/jinja/) to render HTML pages. Also, we need to make sure that we have a templates folder inside our app directory, because by default flask will look inside **templates** directory for all your .html files.

```python
from flask import Flask, render_template

@app.route('/')
def index():
    return render_template('index.html')
```

**Recommended files:** [2_templates.py](https://github.com/Ankur-singh/flask_demo/blob/master/2_templates.py) & [index.html](https://github.com/Ankur-singh/flask_demo/blob/master/templates/index.html)


##### - **Passing values to the front-end**
We can pass values to the front-end by simply passing them as arguments to the `render_template()` along side the .html page that is to be rendered.

You can access the variable by using `{{ <variable_name> }}` in you .html file

```python
@app.route('/heading')
def heading():
    heading = 'Heading from python'
    return render_template('index2.html', heading = heading)
```

**Recommended files:** [2_templates.py](https://github.com/Ankur-singh/flask_demo/blob/master/2_templates.py) & [index2.html](https://github.com/Ankur-singh/flask_demo/blob/master/templates/index2.html)


##### - **Extending base template**
In a website, every page has a lot of redundant content. As a programmer you should always try your best to avoid redundant code. Jinja makes you life pretty easy.

```python
@app.route('/extend')
def jinja_extend():
    return render_template('extend.html')
```

**Recommended files:** [2_templates.py](https://github.com/Ankur-singh/flask_demo/blob/master/2_templates.py) & [extend.html](https://github.com/Ankur-singh/flask_demo/blob/master/templates/extend.html)


##### - **Lists in jinja**
Jinja also provides support for collections in python. So, you can pass a list, string, tuple or dictionary to the front-end and jinja will take care of it.

```python
@app.route('/list')
def jinja_list():
    names = ['Rahul', 'Gandhi', 'Virat', 'Conor']
    return render_template('list.html', names = names)
```

**Recommended files:** [2_templates.py](https://github.com/Ankur-singh/flask_demo/blob/master/2_templates.py) & [list.html](https://github.com/Ankur-singh/flask_demo/blob/master/templates/list.html)


##### - **Conditions in jinja**
Jinja also supports conditions. It can show some HTML code if condition is true and other HTML code if condition is false. This would come in handy in a lot of places.

```python
@app.route('/condition')
def jinja_conditions():
    day = int(np.random.rand(1) * 10) % 2
    return render_template('condition.html', text = day)
```

**Recommended files:** [2_templates.py](https://github.com/Ankur-singh/flask_demo/blob/master/2_templates.py) & [condition.html](https://github.com/Ankur-singh/flask_demo/blob/master/templates/condition.html)

#### Deploying app using flask

Its pretty easy to convert any program (simple program) into a flask app. Just follow these steps.
- Make your code modular
Have all your code organised in form of functions. Try calling these functions from other python file, to see if you have not missed anything. To better understand what I mean by "make your code modular" have a look at [prime.py](https://github.com/Ankur-singh/flask_demo/blob/master/prime.py) & [prime_new.py](https://github.com/Ankur-singh/flask_demo/blob/master/prime_new.py)

- Input & Output
Until this point your program takes input from user and displays the output in the prompt. Its all standard input/output. When you are creating flask app, you input come from a webform and your output is displayed in the browser.
Please have a look at [3_prime.py](https://github.com/Ankur-singh/flask_demo/blob/master/3_prime.py)

#### Deploying ML models using flask

When it comes to deplying ML model, it not very straight forward as creating a function. The challenge here is that machine learning models take very large time to train. Directly converting you code into a function means everytime the URL is accessed, a new model is trained from scratch. If your model takes 20 mins to train means it will take more that 20 mins to respond, every single time. 

The way around this problems is to save the trained model (in-case of DL, you save the model weights) and load the model once when your start the flask app. Then call `model.predict()` every time a request is received. 

There another important thing that is to be taken care off, INPUTs. For normal programs the transformation that you perform on the inputs are not dependent on the data. But in machine learning, all the transformation have some values that are calculated during the training phrase. You will need these values for transformation during the testing phrase. For small projects with very simple dataset and fewer transformations, its pretty easy to save and use these values later. But most of the time, even a simple dataset like "titanic" can require multiple transformations. Hence, instead of manually storing and loading these values is not recommended because it can get pretty messy, very soon. 

Its recommended to use sklearn transformers because of its easy to use APIs. Also, you can easily pickle these objects and use them later. Always remember, in machine learning and deep learning projects, use the same pre-processing for training, testing & deployment. ML/DL models are susceptible to pre-processing and transformations. So, make sure you use libraries, tools & mechanisms that makes it super easy to reproduce pre-processing & transformation. 

**Recommended files:** [iris.py](https://github.com/Ankur-singh/flask_demo/blob/master/iris.py), [iris_new.py](https://github.com/Ankur-singh/flask_demo/blob/master/iris_new.py) & [final_ml_flask.py](https://github.com/Ankur-singh/flask_demo/blob/master/final_ml_flask.py)


Hope you got a glimse of how all these moving parts fit in together. Please refer the files and also pay close attention to the changes that I made. Thank you so much for reading it till the end. Here is a gift for you, list of some really good resources for flask:
- https://hackersandslackers.com/series/building-flask-apps
- https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
