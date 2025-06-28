#  Hyperparameter Tuning – Deep Dive

### Focus: Grid Search CV vs Randomized Search CV

---

##  1. What is Grid Search CV?

`GridSearchCV` is an **exhaustive search** technique that **tests all possible combinations** of hyperparameters from a defined grid and evaluates them using **cross-validation**.

---

###  Analogy: Menu Combinations

> Imagine you’re choosing a combo meal with:
>
> * 3 types of burgers
> * 3 types of drinks
> * 2 sides
>
> Grid search tries **every possible combo**: 3 × 3 × 2 = 18 meals.
> Even if some combos are obviously bad, Grid Search **tests them all**.

---

###  How It Works

1. You specify a set of hyperparameter values to try.
2. It builds a **cartesian product** (all combinations).
3. For each combination:

   * Performs **K-fold cross-validation**
   * Calculates the **average score**
4. Returns the combination with the **best performance**

---

###  Advantages

* **Thorough** and guaranteed to find the best model *in the grid*
* Works well with **small search spaces**

---

###  Disadvantages

* **Time-consuming** for large grids
* Can waste time testing **irrelevant combinations**

---

###  Code Example: GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Define model and parameter grid
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}

# Grid Search with 5-fold CV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

##  2. What is Randomized Search CV?

`RandomizedSearchCV` is a **stochastic optimization technique** that randomly selects a **subset of combinations** from the hyperparameter space and evaluates each one.

---

###  Analogy: Lucky Dip in a Hat

> You put 100 balls (hyperparameter combos) in a hat.
> Instead of trying them all (like Grid Search), you randomly **pick 10**, hoping one of them is a winner.

---

###  How It Works

1. You define **ranges/distributions** for each hyperparameter.
2. You set the **number of iterations** (combinations to try).
3. It **randomly samples** parameter combinations.
4. Performs **cross-validation** on each, then returns the best one.

---

###  Advantages

* **Much faster** with large hyperparameter spaces
* Can **discover better solutions** when you don’t know which ranges are ideal

---

###  Disadvantages

* Results can vary based on **random seed**
* No guarantee of finding the best combo
* May still try **bad combinations**

---

###  Code Example: RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint

# Load data
X, y = load_iris(return_X_y=True)

# Define model
model = RandomForestClassifier()

# Define distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 10)
}

# Randomized Search with 10 trials
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

---

##  3. When Should You Use Each?

| Scenario                                                     | Use Grid Search If… | Use Randomized Search If… |
| ------------------------------------------------------------ | ------------------- | ------------------------- |
| You have **few parameters** and **few values per parameter** | ✅                   | ❌                         |
| You have **many parameters or wide value ranges**            | ❌                   | ✅                         |
| You need a **guaranteed best** from known values             | ✅                   | ❌                         |
| You care about **speed over perfection**                     | ❌                   | ✅                         |
| You **don’t know** the best value ranges yet                 | ❌                   | ✅                         |
| You can afford long computation                              | ✅                   | ❌                         |
| You need **quick tuning in prototyping**                     | ❌                   | ✅                         |

---

## 🧾 4. Comparison Table: Grid Search CV vs Randomized Search CV

| Feature                         | Grid Search CV          | Randomized Search CV    |
| ------------------------------- | ----------------------- | ----------------------- |
| Search Type                     | Exhaustive (all combos) | Random sampling         |
| Speed                           | Slower                  | Faster                  |
| Best Model Guarantee            | Yes (within grid)       | No                      |
| Use Case                        | Small parameter spaces  | Large or unknown spaces |
| Custom Distributions            | ❌ (only exact values)   | ✅ (use `scipy.stats`)   |
| Risk of Overfitting to CV folds | Moderate                | Lower (fewer combos)    |
| Easily Parallelizable           | ✅                       | ✅                       |
| Reproducibility                 | ✅                       | ✅ with fixed seed       |

---

##  5. Final Analogy Recap

| Analogy                       | Concept                             |
| ----------------------------- | ----------------------------------- |
| Trying every meal combo       | Grid Search CV                      |
| Sampling a few pizzas         | Randomized Search CV                |
| Ingredients vs Recipe Results | Hyperparameters vs Model Parameters |

---

##  Bonus Tips

* Use **GridSearchCV** when you’re refining or benchmarking models
* Use **RandomizedSearchCV** early to **explore quickly**
* Always combine either with **cross-validation**
* Tune fewer parameters at a time for better control

---

