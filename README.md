# HateSpeech-Detection

🕵️ Introduction
There is tonnes of hate speech being posted everyday on social media by different users. Facebook runs its hate speech detection algorithm and actively removes content which is hateful. The objective is to build a machine learning model that classifies a piece of text as hate speech or not. An example of a hateful sentence is

    “​ I don’t know how much more I can take! 45 is a compulsive liar! #Trump30Hours #TrumpIsATraitor ” .
The problem statement is that we are given a piece of text, which we need to classify into hate speech or not hate speech. It is a binary classification problem with labels “HOF”(0) denoting hate speech and “NOT”(1) denoting non hateful sentences. Numbers in brackets denote the class label.

💾 Dataset
For training, you will be provided a csv file containing 2 parameters -

Text -text field containing the tweet/ comment and a label
Label containg the label 0 for HOF and 1 for NOT
