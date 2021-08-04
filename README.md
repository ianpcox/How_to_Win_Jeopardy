# How_to_Win_Jeopardy
The goal of this project was to use hypothesis testing to recommend how to best prepare for the popular trivia gameshow <i>Jeopardy</i> with the expected outcome of earning the most money.  

There were two primary areas of analysis: 
* How often a given answer can be used for a question, and 
* How often questions were repeated.  
 
A chi-squared test is used to narrow down the questions into two categories: 
* Low Value, and 
* High Value.

These measures of value are combined with the two primary areas of analysis above to determine whether there was significant difference in usage.  It was determined that since the frequencies were all under 5, the chi-squared test wasn't as valid, b ut signals that it would be better to run the test with terms that have higher frequencies.
