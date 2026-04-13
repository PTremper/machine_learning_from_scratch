# Random Forest

## Core idea

A random forest combines many decision trees.

Each tree makes a prediction, and the forest takes the **majority vote**.

---

## How it learns

For each tree:

1. **Create a random dataset**

   * Sample from the original data *with replacement*
   * Some samples appear multiple times, some not at all

2. **Train a decision tree**

Repeat this process for many trees.

---

## How it predicts

For a new sample:

1. Each tree makes a prediction
2. Collect all predictions
3. Return the most common one (**majority vote**)

---

## Key ideas

* Trees are **diverse**, trees are *weak experts*
  * Each tree sees **slightly different data** --> *each expert sees different cases*
  * Each tree grows splitting **randomized features** --> *each expert thinks in a different way*
* Combining them makes predictions more stable

---

## Why it works

A single tree can overfit (memorize the data).

A random forest reduces this by:

* averaging over many trees
* reducing reliance on any single split or feature
