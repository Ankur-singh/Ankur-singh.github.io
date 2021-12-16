---
comments: true
title: SQLite - Light & Powerful
tags: [databases, SQL]
style: border
color: danger
description: This blog is a quick overview of SQLite. It’s a very light database. The complete database is just a single binary file.
 
---


![](https://cdn-images-1.medium.com/max/1200/1*U0TVDqlBF69q8i3Zz6-fKA.jpeg)

Photo by [Jordan Harrison](https://unsplash.com/@jordanharrison?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/data-center?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)


This blog is a quick overview of SQLite. It’s a very light database. The complete database is just a single binary file. Unlike other databases, which uses complex directory structure & multiple files to store the database on the disk. Also, SQLite requires no complex setup. Its both open-source & cross-platform.

Because of its simplicity, I love working with it. But I am not writing this blog because of its simplicity. I am writing this blog, because it’s equally powerful.

Its a database, but it’s very different from all other enterprise grade RDBMS. The below text is copied from the official docs. Its captures the gist very well.

> SQLite is not directly comparable to client/server SQL database engines such as MySQL, Oracle, PostgreSQL, or SQL Server since SQLite is trying to solve a different problem.

> Client/server SQL database engines strive to implement a shared repository of enterprise data. They emphasize scalability, concurrency, centralization, and control. SQLite strives to provide local data storage for individual applications and devices. SQLite emphasizes economy, efficiency, reliability, independence, and simplicity.

> SQLite does not compete with client/server databases. SQLite competes with `fopen()`.

I highly recommend you to read [Appropriate Uses of SQLite](https://sqlite.org/whentouse.html). It describes situations where SQLite is an appropriate database engine to use versus situations where a client/server database engine might be a better choice.

Before we jump into SQLite and its details. Here are some prerequisites, just to make sure that we are on the same page.

### Prerequisites

*   Make sure you have SQLite up and running in your system.
*   All the database files used in this blog can be downloaded from [here](https://drive.google.com/drive/folders/1RCWx4q-53DVDlC9yYGWHM6fCZmWpi4kJ?usp=sharing).
*   I am assuming your are comfortable with terms like RDBMS, SQL, SQLite, servers, etc.

If answer to any of the above is NO, then I will highly recommend you to read [this blog](https://sanjaysodha2607.medium.com/databases-and-quick-overview-of-sqlite-5b7d4f8f6174) first. Great, I think we are good to start.

### Loading the database

You can load any SQLite database by running the following command:

    $ sqlite3 enrollments.db

The above command will load `enrollments.db` into the RAM. If `enrollments.db` does not exist in the current working directory, then it will create an empty database and load it.

As soon as you load the database, you should look at the schema. In SQLite, we use `.schema` command to print database schema. Database schema shows the following information:

*   All the tables that are present in the database
*   All the fields in each table &
*   Relation between different tables in the database

In `enrollments.db` there is only one table `enrollments` and this is how the schema looks.

![](https://cdn-images-1.medium.com/max/800/1*8pZMPKDhuXnlX7sLRljs3g.jpeg)

Looking at the schema, we know there are 5 columns. `email`, `name`, `course`, `joining_date` are all TEXT, and `duration` is INTEGER. Actually, the same statement was used to create the `enrollments` table.

This is very thoughtful, because now you don’t have to remember two different syntax. If you know how to create a table, then you also know how to read the schema. Such small things matter a lot when working with CLI tools.

* * *

### Dot commands

**Note:** All the commands that start with a dot (“ . “) are specific to SQLite. These commands will not work in other RDBMS. Also, _dot_ commands don’t need “ ; “ at the end.

You can use `.help` to see all the available commands. Not all of them are equally useful. So, here is a list of commands that I use most frequently.

*   `.table` : List names of tables
*   `.schema` : Show the CREATE statements or table schema
*   `.import` : Import data from FILE into TABLE
*   `.open` : Open a database
*   `.read` : Execute SQL query in FILE
*   `.show` : Show the current values for various settings
*   `.exit` : Exit SQLite prompt

To read more about any _dot_ command you can simply use `.help mode` . Remember, _dot_ is part of the syntax not the commands itself. Hence, you don't have to use _dot_ again in front of `mode`.

* * *

### Selecting data

Before we move ahead, ponder for a minute. What is the most frequently used operation (that you perform) when using a GUI based tool like MS Excel? You guessed it, the answer is **Selecting!** You select the data first and then select the operation from the menu bar at the top. You select the cell you want to update or delete. In shot, you select before every operation.

In GUI based applications, its pretty obvious. But in RDBMS, we have to write SQL queries to select data. Knowing how to select data is the most fundamental operation. Before you can doing anything, you will have to select your data. Let’s get started!

### `select` clause

The `select` clause helps you to select column(s).

    -- Selecting single columnsqlite> select email from enrollments;

    -- Selecting multiple columnssqlite> select name, email from enrollments;

    -- Select all columnssqlite> select * from enrollments;

**Note** : Relations databases have multiple tables. So, all the queries will require you to mention _table\_name_ (here, `enrollments`).

Pretty simple, right?

But when you select multiple columns, the output is not very readable. Run the following commands:

    sqlite> .mode columnssqlite> .header onsqlite> .width 20 10 18 12 8sqlite> select * from enrollments;

`.mode`, `.header`, & `.width` are all SQLite commands. Use `.help mode`, `.help header`, & `.help width` to learn more about them.

You can use `limit` clause to specify the number of records to return. Here, it’s a very small database. `limit` clause is extremely useful when working with large tables with thousands of records. Because returning a large number of records can impact performance.

    -- to print only top 10 recordssqlite> select * from enrollments limit 10;

### `where` clause

The `where` clause helps you to select record(s)

    -- Select all the records where the condition/expression evaluates to TRUEsqlite> select * from enrollments where condition/expression;

SQL supports many of different types of conditions. They are listed as follows:

    -- Comparison operators (=, !=, <=, >=, <, >)sqlite> select * from enrollments where duration > 6;

    -- Membership operator; 'in' & 'not in'sqlite> select * from enrollments where course in ('Python', 'Machine Learning');

    -- 'between' operator; both the lower & the upper limits are includedsqlite> select * from enrollments where joining_date between '2019-10-11' and '2020-10-11';

    -- Pattern matching; 'like' & 'glob' operatorsqlite> select * from enrollments where course like '%learning%';

It looks like a lot, but it’s as good as reading & writing English. You will get use to it very quickly, once you start writing some queries. Below are a few details that you need to understand to use these conditions effectively!

*   SQLite uses single quotes ( ‘ ‘ ) to represent strings. Hence, in all the above queries, strings are inside single quotes.
*   In SQLite, “equal to” operator is `=` (not `==`). Many programming languages uses `==` as "equal to" operator.
*   In SQLite, strings are case-sensitive i.e. ‘Python’ & ‘python’ are not equal.
*   The GLOB operator is similar to LIKE but uses the Unix file globbing syntax for its wildcards. LIKE uses `%` & GLOB uses `*` for wildcard.
*   Also, GLOB is case sensitive, unlike LIKE.

SQLite supports a very comprehensive set of operators to from conditions. On top of all this, you can combine these conditions using `and`, `or` & `not` operators. This allows you to from even complex queries. For example;

    sqlite> select * from enrolments where course = 'Python' and duration > 6 and not joining_date between '2018-10-11' and '2019-10-11';

This is amazing. To select a subset of data, we can use `select` & `where` clause. `select` clause allows us to **select columns** and `where` clause allows us to **select rows.** This is a very powerful notation. Once you master it, selecting data will be a piece of cake!

Before reading any further, I would recommend you to try writing queries for the following questions:

### Practice Questions:

1.  Select name, email, & course for all the records.
2.  Select all the student names who enrolled for python after 2019.
3.  Select all the records of students who enrolled for courses which have “learning” in their name.
4.  Select all the students who completed the course in 5, 7 or 9 weeks.

### Performing Operations

There are many different operations that you can do once you have selected your data. Let’s explore a few operations, one by one:

### `order by` clause

You can use `order by` clause for sorting you records based on one or more columns in ascending or descending order.

    -- ascending order based on joining datesqlite> select * from enrollments order by joining_date;

    -- sort the records based on multiple columnssqlite> select * from enrollments order by joining_date desc, name;

You can use `asc` or `desc` keywords after _column\_name_ to specify sorting order. If nothing is specified then default `asc` order is assumed.

In Case of multiple columns, a comma (“ , “) is used to separate _columns\_name._

### SQLite functions

SQLite has many built in functions. They are very similar to functions in other programming languages.

Here are some functions for string manipulation.

*   `upper` - Return a copy of a string with all of the characters converted to uppercase.
*   `lower` - Return a copy of a string with all of the characters converted to lowercase.
*   `length` - Return the number of characters in a string or the number of bytes in a BLOB.
*   `trim` - Return a copy of a string that has specified characters removed from the beginning and the end of a string.

Here are some aggregate functions

*   `avg` - Return the average value of non-null values (in the group).
*   `count` - Return the number of non-null values (in the group). The count(\*) function (with no arguments) returns the total number of rows.
*   `min` , `max` & `sum` - Return the min, max & sum of non-null values (in the group).

You can read more about [aggregate functions from the docs](https://sqlite.org/lang_aggfunc.html).

This is how you use SQLITE functions:

    -- uppercase all the namessqlite> select upper(name) from enrollments;

    -- avg, min & max duration of all the python coursessqlite> select avg(duration), min(duration), max(duration) from enrollments where course='Python';

You can read more about SQLite functions, [here](https://www.sqlitetutorial.net/sqlite-functions/).

There is also a `distinct` clause which can be used just like SQLite functions to return only the distinct records/values. Here is an example:

    -- all the different courses at aiadventruessqlite> select distinct(course) from enrolments;

### `group by` clause

The `group by` clause divides the records into groups based on the values of one or more columns. These groups become really handy when working with aggregate functions. In fact, aggregate functions are meant to be used with groups. They calculate aggregate values for each group.

In the above examples, we never used `group by`, so all the records were considered in one group.

    -- returns the first record from each groupsqlite> select * from enrollments group by course;

    -- returns min, max & avg duration for each coursesqlite> select min(course), max(course), avg(course) from enrollments group by course;

`group by` clause is really powerful once you understand it. So, I'll recommend you to try running a few queries with `group by` clause in it.

In this section, we learnt `order by` clause, SQLite functions, & `group by` clause. Before this we learnt `select` clause & `where` clause. There are many different clauses, operators & functions to learn and remember. But with a little practice, SQL becomes very obvious. So, here are some more practice questions.

### Practice Questions:

1.  Select all the students who enrolled for python course. Sort them based on joining data (latest at the top), and if there are multiple people who joined on the same date, then order them alphabetically based on their names.
2.  What is the average time required to complete the machine learning course?
3.  List all the different courses that [aiadventures](http://www.aiadventures.in) offer.
4.  What is the average, minimum & maximum time required for all the courses? Also, show the course names.

### CRUD (Create, Read, Update, Delete)

In computer science, CRUD is a very famous acronym for create, read, update and delete. These are the four basic functions of persistent storage. We already know how to read records using `select` & `where` clause. Lets see how we can create, update & delete records.

### `insert into` clause — Create new record

Inserting a new record is pretty easy. Let’s add some new records:

    -- inserting single recordsqlite> insert into enrollments values ('akshata@gmail.com', 'akshata', 'Python', '2020-12-06', '');

    -- inserting multiple recordssqlite> insert into enrollments values ('lehanshu@gmail.com', 'lehanshu', 'Data Science', '2020-12-09', ''), ('shradha@gmail.com', 'shradha', 'Data Science', '2020-12-09', '');

You just specify the _table\_name_ and the values that are to be inserted.

Always remember, you will have to specify the _table\_name_ in almost every query, because the database can have multiple tables.

### `update` clause — Update existing record

To update any record, first you will have to select it. We will use `where` clause for selecting records. Then, you will have to select the values that are to be updated using `set` keyword. Here is query in action:

    -- update duration for all the records where duration is ''sqlite> update enrollments set duration=6 where duration = '';

    -- update duration & course for all the records where duration is ''sqlite> update enrollments set duration=6, course='Machine Learning' where duration = '';

The _duration_ for all the records that are selecting using `where` clause will be updated to _6_. You can also update multiple values, as shown above in the second query.

### `delete` clause — Delete records

    -- Delete all the records where name is 'ankur'sqlite> select * from enrolments where name='ankur';sqlite> delete from enrolments where name='ankur';

It’s very important, to verify your selection before `update` or `delete` clause. Because there is no UNDO operation.

So, always run `select *` first and then replace it with `delete` . In SQL, you cannot delete a single value, you will have to delete the complete record. Hence, there is not need for column selection i.e `*` operator with `delete` clause.

We have come a long way, let’s revise everything. We can select records using `select` and `where` clause. Once we have selected the records, we can perform operations on them, calculate aggregate values, & update/delete them. These operations might look very basic, but they are like LEGO blocks. You can combine them in any fashion to meet your requirement.

### Practice Questions:

1.  Try inserting a new record. Imagine you are joining a course at aiadventures. Keep the duration field empty.
2.  Update the duration for the record that you entered in the last question.
3.  Delete any record.

The real power of RDBMS systems is unleashed when start combining tables. Let’s discuss it next!

### Combining tables — `join` clause

**Note:** Hence forth, we will be using `movies.db` . You can load it by running `sqlite3 movies.db` in your terminal window. Don't forget to have a look at the database schema.

Combining tables is pretty simple & straight forward. To join two tables, you need a common key. Let’s look an example.

`movies.db` has 5 different tables, `movies` and `ratings` are the two tables that we are interested in (for now). This is how the schema looks for `movies` & `ratings` table.

![](https://cdn-images-1.medium.com/max/800/1*kDFvX_AkTENjgnpVHdlJSw.jpeg)

Say we want to print the movie titles & their ratings. This data is not present in any single table. Movie titles are present in `movies` table and ratings are present in `ratings` table. We can combine both the tables using `join` clause, based on `movie_id` column in `ratings` table & `id` column in `movies` table.

    -- combining movies and ratings tablesqlite> select title, rating from movies join ratings on movies.id = ratings.movie_id;

Notice, after _from_ we always write the _table\_name;_ but here we don’t have a single table. We had to create a single table by combining `movies` & `ratings` table. Hence, only the _from_ part is changed. You can still use `where` , `group by` , `order by` , etc. clause as usual.

Combining tables is a huge topic. There are many different types of join (cross, inner, outer). SQLite only supports CROSS JOIN, LEFT OUTER JOIN, INNER JOIN. But in-depth discussion is not the focus of this blog. If curious, you can read more about all the different types of joins in SQLite and their applications, [here](https://www.techonthenet.com/sqlite/joins.php).

### Final words

Until this point, it looks same as excel. I will say, even less powerful than excel. But the best things about RDBMS is, they provides interfaces for different languages like python, java, etc. So you can use programming languages to execute SQL queries.

This is of immense importance for server applications that are running 24x7. RDBMS are not for day-to-day use by general public, we have MS Excel for it.

Whenever learning a new tool, its equally important to understand the limitations. Having a clear understanding of the strengths and weakness is important to master any tool. Just like everything else in life, SQLite (or RDBMS in general) has its own **limitations**:

*   SQL is a query language. So, doing anything else is pretty difficult.
*   SQL is structured, so doing things that would require you change the schema are difficult & not very obvious.
*   SQL is row first, so columns operations are pretty difficult to perform. Like adding 10 to a particular column of all the selected records.

Being from a data science background, I use pandas very extensively. So, I was very tempted to compare SQL & pandas. I guess many more people would also be tempted. So, here are some resources that I found.

*   [https://datascience.stackexchange.com/questions/34357/why-do-people-prefer-pandas-to-sql](https://datascience.stackexchange.com/questions/34357/why-do-people-prefer-pandas-to-sql)
*   [https://www.quora.com/In-what-situations-should-you-use-SQL-instead-of-Pandas-as-a-data-scientist](https://www.quora.com/In-what-situations-should-you-use-SQL-instead-of-Pandas-as-a-data-scientist)
*   [https://pandas.pydata.org/docs/getting\_started/comparison/comparison\_with\_sql.html](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)

Remember, no one tool is better than the other. Both have their own advantages & disadvantages. They were built with different intentions, to solve different types of problems.

### Resources

*   Complementary [Git repo](https://github.com/Ankur-singh/sqlite_workshop)
*   For SQLite tutorials, visit [https://www.sqlitetutorial.net/](https://www.sqlitetutorial.net/)
*   Refer [SQLite Quick Guide](https://www.tutorialspoint.com/sqlite/sqlite_quick_guide.htm) for SQLite concepts and syntax
*   See [this SQL keywords reference](https://www.w3schools.com/sql/sql_ref_keywords.asp) for some SQL syntax that may be helpful!

### For more practice

*   [CS50x course](https://cs50.harvard.edu/x/2020/psets/7/)
*   [HackerRank](https://www.hackerrank.com/domains/sql)

### To learn next . . .

The above topics are enough for a beginner or a part-time SQL user, but this is not all. There is so much more to SQL & SQLite. Below is a list of things to explore next, if you decide to become a SQL ninja. Google each of the following, and read about them in detail.

*   Sub-queries
*   Views
*   Indexes
*   Triggers
*   Transactions

Finally, this is how you print something in SQL. 😁

    sqlite> select 'thank you!';thank you!

Hope you had a wonderful time reading it. Next blog will discuss “How to run SQL query using/from python?”. So stay tuned! Also, don’t forget to clap and share.
