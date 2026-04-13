# Decision Tree (Multi-Class Classifier)

## Core idea

A decision tree makes predictions by asking simple yes/no questions like:

> “Is this value smaller than some number?”

Each question splits the data into two groups.

---

## How it learns

When growing the tree, at each node, the algorithm:

1. Picks a subset of features (randomly)
2. Tries all possible splits (feature + threshold)
3. Chooses the split with the **highest information gain (entropy decrease)** in the labels
4. Recursively repeats this for the resulting subsets

Eventually, it stops and assigns a class label to each leaf.

Stop when:

* Only one class label left
* The sample size gets too small
* The tree gets too deep

---

## How it predicts

To classify a new sample:

* Start at the top
* Each node is a question to the data (left/right)
* Stop when reaching a leaf → that’s the prediction

---

## Key ideas

* Keep splitting the data into smaller groups using simple rules
* Each split uses only **one feature at a time**
* The goal is to make each group as homogenous as possible

---

## Why it works

* Breaks a complex problem into many simple decisions
* Can model non-linear patterns through repeated splits
* Easy to follow and interpret the final decisions
