---
title: Complete workflow of a Machine Learning project
tags: [Machine Learning, Sklearn, Preprocessing, fast.ai]
style: fill
color: secondary
description: Learn to manage your next Machine Learning Project better.
---

Current practitioner of Machine Learning will know about the difficulties associated in handling a ML project. Sometimes, it might appear daunting to an absolute beginner in the field (as a ML project has a lot of moving parts, all entangled together to make a single thing work). In this article, I have broken down ‘the Complete process’ into small steps such that you can master them individually.

The complete article is inspired by Jeremy Howard’s [fast.ai](https://www.fast.ai/) course. A huge shout out to [Jeremy ](https://twitter.com/jeremyphoward)and [Rachel](https://twitter.com/math_rachel) for their amazing course. I did the course, it was absolutely enlightening. Highly Recommended! With all this newly acquired knowledge, I had a strong urge to share it and I ended up writing this article.

I have also create kaggle kernels ([Part-1](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-1) and [Part-2](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-2)). High recommend if you want to get a feel of how everything works in Code.

Lets get started. How to approach a ML project? Answer:

## 1. Defining the problem statement

The key thing to define a machine problem is to identify what are the independent and dependent variable in the data. The dependent variable is the thing you’re trying to predict.

## 2. Use “Tiny Data” for initial Exploration

Take a tiny subsets of the different csvs to start exploring the types of data. We’ll portion off the first 5 lines of data in our train.csv into its own subset.

    head -5 train.csv > tiny_subset.csv

Our first exploration is to read in tiny_subset.csv with pandas library. We can use the tiny_subset, to figure out what the datatypes are with pandas before reading in the whole table. tiny_subset.csv lets us explore dataset’s features quickly because we can execute code in seconds instead of minutes. For example, We learn the datatypes for each column in tiny_subset.csv are:

    types = {
    'id' : 'int64',
    'item_nbr' : 'int32',
    'store_nbr' : 'int8',
    'unit_sales' : 'float32',
    'onpromotion' : 'object'
    }

From our tiny_subset.csv, we know that we have a ‘dates’ column, so we can tell pandas to parse that column as a date. After our initial exploration, we know what the major data types and features are and can optimize the reading in the whole train file.

    df_all = pd.read_csv('path-to-/train.csv', parse_dates = ['dates'], dtypes = types, infer_datetime_format=True)

After our initial processing, the whole train.csv file now can be loaded in seconds with very little data tweaking or corrections.

Assigning proper data dtypes to your columns will further save you a lot of memory and computation. Use *reduce_men_usage()* for this.

## 3. Data Preprocessing / Cleaning

At this stage, your focus is to transform the data so that it can be feed as input to a Machine Learning model. It involves

* Identifying categorical and Numerical features

* Encoding categorical features

* Adding an Extra column to represent whether a numerical value is present or not

* Normalize numerical values (depending on the model)

* Extract features from timestamps (if present)

* Try making some new explicit features (like *has_cabin*, *family_size* & *is_alone *in titanic dataset)

By this time you start to get an idea about your data. So, you can start with some data analysis like identifying the outliers, understanding the trends using plots, etc. Also, if you are an expert in the domain, try integrating your expertise with the data available.
> **Note:** Keep Using the *Tiny subset of data* where execution of a programming process is completed in around 10 seconds so you know that the part of the pipeline you’re building works. You need to iterate and tweak the pipeline and model quickly. If you want to test the whole data pipeline from data ingestion to validation, use a larger subset of training data, but you don’t have to use all training data just to know that your end-to-end pipeline works.

After optimizing and cleaning the dataset we’ll read in whole dataset with pandas, and use *describe(include=’all’)* to get summary stats about the data.

After reading in our csv data as dataframes and processing it, we’ll use **feather format** to write and read our processed dataframes to disk. Feather format reads/writes to disk as fast as reading/writing to memory.

### **Speeding up Things**

We can time your functions with *%time* in front of any function in your jupyter notebook and can measure how long it takes a function to run. In addition you can run *%prun* in front of any line of code in your notebook which runs a profiler that examines all the lines of code under the hood of that code statement.

For example, we can convert dataframe to **np.array float32** before given to RF. Random Forest will change the dataframe to numpy array anyway. If you convert the dataframe to a numpy array once yourself, the model doesn’t have to do that each time. If you want to run multiple models, you’ll save that conversion time for each model. You will find out this only when you run *%prun*.

Profiling code is highly under-appreciated by data scientists. Although you didn’t write the sklearn Random Forest library, you can learned to make it run twice as fast. It’s worth exploring and experimenting on how to use profiler outputs.

## 4. Keep your model analysis fast and light at first

You don’t have to call the full dataset for random forest analysis. You only needs as much of the data that will show the types of relationship that are involved. Only when you’re comfortable with what’s in the sample, you can go on apply the same to the complete data.

Once you get your preliminary results about your scores from the model, you can start playing with the different parameters. Here’s is a pretty clear walk through of the parameters of the [Random Forest Regressor](https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/). You can tweak any of the parameters with the goal of getting better scores.

Once you’re done iterating, then you can train the model on the all the training data overnight. If you have access to more than one gpu, you can train one model on the gpu while you iterate on another model (maybe one with a different architecture or the same model with different hyperparameters)

## 5. Model Interpretation (aka Model do more than prediction)

Machine learning model’s provide more information than predictions. In addition, to wanting to know a prediction, we want to know how confident we can be about those predictions.

### a) Confidence Intervals based on standard deviation:

If a trained random forest model sees a novel datapoint, the tree/estimator that assesses it will treat it differently than the information it has already seen. It may end up on an outlier branch of the decision tree. If we only use confidence intervals based on averages, the novel data point won’t be judged accurately. Instead we can use, standard deviation of the trees’ prediction to provide confidence levels for the predictions. If standard deviation is high, this means each tree is giving a very different estimates of this rows’ prediction. If the observation is judged to be a very common kind of row, we’ll have tighter/smaller standard deviation score.

Confidence intervals can help you learn about which groups or features the model may not be confident about. sklearn doesn’t have such a library that outputs the standard deviation of trees’ prediction, but we can make one.

Taking the standard deviation of the trees’ prediction can help us explore the unknowns about the data set. After plotting the trees’ predictions and confidence intervals with the difference features, we can do exploratory data analysis on what’s important even when we don’t know what the features is exactly. The bar charts of the confidence levels gives us an intuition behind how to apply different feature groupings and we can identify which features contributed to low prediction accuracy. Are there some groups that the model isn’t confident about? This confidence value could also be used in as an end product, like in loan application. We could judge yes/no on giving a borrower a loan, and we could also provide a confidence level on whether or not the borrower will pay it back.

### b) Feature importance :

To determine feature importance, you would build a Random Forest as fast as you can. You’re aiming for an accuracy score better than random, but not much more than that. Then you would plot the feature importance of the different fields in the analysis

**Which features matter in random forest ?**

If you plot all the features with their corresponding importance, you’ll find that some columns are important, and some don’t matter at all. Understanding which features are important directs us where we to need to gather domain knowledge. This is the part where you sit down with a client and ask about the key features. You would do exploratory data analysis on key variables: run different plots or significance tests. If the feature reveals itself to be important for prediction and client says it’s not, this could indicate a data leakage.

Feature importance analysis might also find co-linearity in variables, so we might see a few features that are appear important but in fact signal the same thing or a similar trend. You have to be careful with your analysis if you see this happening. One experiment is to start throwing data out of the analysis to see if anything changes in your prediction. We’ll exclude any feature that scored lower than (say ).005 importance. We can create a new dataframe using only these top features; we’ll divide this slimmed dataset into a test and train sets and pass these to a new Random Forest instance. Your accuracy will either decrease only a bit or will increased after throwing out less important features.

Generally, throwing out redundant columns shouldn’t make your model worse; if accuracy goes down, those columns weren’t redundant after all. Tossing out redundant columns also lowers the possibility of co-linearity (aka two columns that may be related to each other). In a random forest, the tree will mistakenly group different features together because they’re similar.

Understanding the important features of a dataset lets us concentrate on what matters and will make our models run faster. By removing low impact variables, we make our feature importance plots clearer, we can trust these features’ importance more.

### c) Surfacing data leakage can be useful:

What is data leakage? — A feature of the data becomes available that was not originally intended when the original data was input or when the dataset released. In other words, there’s information about dataset that you have that the client didn’t have at the time the dataset was created.

This unintended feature can be surfaced during data exploration and interviews with the data stakeholders. For example, Jeremy worked on predicting successful applications for a university grant program, and he found out a single feature–whether the applications were submitted or not, determined whether a grant was was funded. However, he talked and listened to all the people involved in the dataset’s creation. He discovered due to the fact that it was administratively burdensome to log the data around the grant applications, administrators only entered successful grants into database. To make a valid model, this feature needed to be left out of the analysis.

Understanding data leakage is important because either this data leakage feature leads the analyst to make a mistaken conclusion o r to build an inaccurate model. Investigating data leakage takes legwork and exploration that may lie beyond the data in front of you. On the other hand, a data leak can used as additional feature to make a better performing model in some situations (i.e. Kaggle competitions).

## Final Words:

By this time you might have already realize that building a good Machine Learning model / project is some what complex process. It involves a lot back & forth. Hence, data scientists use jupyter notebook because its interactive and make working on machine learning projects much easier. This is the first advice, use jupyter notebook. It might look complex at first but the workflow is really very intuitional. So, practicing it once or twice will make you extremely good at it (second advice).

Do comment if you find it useful or if you have any feedback.
