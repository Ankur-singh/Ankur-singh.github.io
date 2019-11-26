---
title: Text Classification:- ULMfit v/s Logistic Regression
tags: [Machine Learing, Deep Learning, NLP, Fastai]
style: border
color: warning
description: Any guesses, which one is better? ULMfit or Logistic Regression?
#external_url: https://blog.usejournal.com/how-to-undo-your-git-failure-b76e31ecac74
---

Any guesses, which one is better? ULMfit or Logistic Regression?

Made your choice? Hold on to it till the very end of the blog. Talking of the best algorithm, ever heard of the **“No Free Lunch Theorem”** ?

**Sorry!** won’t bore you any more with all these random questions. Lets dive in.

## Dataset:

It is from a [competition ](https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon/)hosted by [Analytics Vidya](https://analyticsvidhya.com). It is a text classification problem. You have a review about a certain drug and your task is to predict the sentiment. Lets look at the data:

![](https://cdn-images-1.medium.com/max/2000/1*ihz5xa7Xdfb2-naG9iL4gQ.jpeg)

* text : some review about a particular drug

* drug : name of the drug

* sentiment (target) : sentiment of the review
> **Note** : For the sake of simplicity, I will only be using the “text” column to make predictions

To download the dataset or to have a look at it yourself, visit [this](https://www.kaggle.com/japkeerat/innoplexusav).

## Fastai’s ULMfit:

People who have already used ULMfit, they know how awesome it is. Others, I highly recommend you to check it out. In simple terms, its a transfer learning approach for NLP.

### Language model:

![](https://cdn-images-1.medium.com/max/2270/1*ULrdVAG5r1b1alaF2omNMg.jpeg)

This does all the necessary pre-processing behind the scene. We can use the data_lm object to fine-tune a pretrained language model. [fast.ai](http://www.fast.ai/) has an English model with an AWD-LSTM architecture available that we can download. We can create a learner object that will directly create a model, download the pretrained weights and be ready for fine-tuning.

### Classifier:

For the classifier, we also pass the vocabulary (mapping from ids to words) that we want to use: this is to ensure that data_clas will use the same dictionary as data_lm.

We now use the data_clas object to build a classifier with our fine-tuned encoder. The learner object can be done in a single line.

![](https://cdn-images-1.medium.com/max/2272/1*gdTzjXvEGyNptJf_ODGKQw.jpeg)

Like a computer vision model, we can then unfreeze the model and fine-tune it.

![](https://cdn-images-1.medium.com/max/2312/1*SvQR-4o-sIexbm9ikqprrQ.jpeg)

After fine-tuning the language model as well as the classifier I got 0.47, f1_score(macro averaging), which I thought was really good until I saw the scores for logistic regression.

## Logistic Regression:

![](https://cdn-images-1.medium.com/max/2330/1*iI7gLXEHgOkpx5msSXo2IA.jpeg)

This is all you have to do. TF-IDF to convert “text” column into 60,000 long vectors and then logistic regression to classify those vectors. It easily gave me 0.72, f1_score(macro averaging).

People curious about the “evaluate_model” function,

![](https://cdn-images-1.medium.com/max/2272/1*_RGgugWeSk4tCNRG-P2G1g.jpeg)

## Conclusion:

There is **NO single best** algorithm for all the problems. This is [NO FREE LUNCH THEOREM](https://chemicalstatistician.wordpress.com/2014/01/24/machine-learning-lesson-of-the-day-the-no-free-lunch-theorem/). Here, Logistic regression is easy able to beat ULMfit but in a lot of other cases you will see the opposite.

### Important words to google next:

* ULMfit & Fastai (transfer learning in NLP)

* NO FREE LUNCH THEOREM

* Language Models

I have made the complete code available [here ](https://www.kaggle.com/ankursingh12/drug-sentiment-analysis/). You are free to use any part of the code.

If you found this article helpful, please share. It will really encourage me to write more such helpful articles. Also, don’t forget to upvote the [kaggle kernel](https://www.kaggle.com/ankursingh12/drug-sentiment-analysis/).

We will try to learn why and how Logistic Regression is able to performing better than ULMfit, in the next post which will be on “interpreting text classifiers”. So stay tuned & make sure you *follow me* for regular updates. Until next time, Keep Learning!

### References:

* [https://docs.fast.ai/text.html](https://docs.fast.ai/text.html)
