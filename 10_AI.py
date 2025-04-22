# import markovify

# # imagine we have a sentence like this:
# # "The quick brown fox jumps over the lazy dog. The dog barks at the fox."
# # We want to generate new sentences that are similar in style to this one.
# # We can use the Markovify library to do this.
# # How it works
# # 1. It takes a text and builds a Markov chain model from it.
# # markov chains are a type of stochastic model that predicts the next state based on the current state.
# # 2. It generates new sentences by randomly selecting words based on the probabilities of their occurrence in the original text.


# # 1. Load your text (could be a file or just a big string)
# with open("sample.txt", mode="r+",encoding="UTF-8") as f:
#     text = f.read().lower()

# # 2. Build the model
# #    - state_size=1 is a first‑order chain (one word → next word)
# #    - state_size=2 is a second‑order chain (two words → next word)
# model = markovify.Text(text, state_size=2)

# # 3. Generate sentences
# # print(model.make_sentence())               # one full-ish sentence
# # print(model.make_short_sentence(30))       # ≤ 50 characters
# print(model.make_sentence_with_start("look at",strict=False))  # start with a specific word








# # Įdiekite biblioteką, jei jos dar neturite:
# # pip install efficient-apriori

# from efficient_apriori import apriori

# # 1. Duomenų paruošimas: Kiekviena transakcija – pirkinių krepšelis su įsigytomis prekėmis.
# transactions = [
#     ['duona', 'sviestas', 'sūris'],
#     ['duona', 'miltai'],
#     ['sviestas', 'pienas', 'kava'],
#     ['duona', 'sviestas', 'pienas'],
#     ['sūris', 'pienas'],
#     ['duona', 'sūris', 'sviestas']
# ]

# # 2. Dažnių derinių suradimas:
# # Nustatome, kad derinys turi pasirodyti bent 30 % transakcijų.
# # min_support = 0.3, t.y. 30 % visų transakcijų.
# # Be to, analizuojame taisykles, kurių confidence yra bent 70 %.
# min_support = 0.3
# min_confidence = 0.7

# # Apriori funkcija grąžins du dalykus:
# # - itemsets: Dažni deriniai, surasti pagal minimalų support.
# # - rules: Asociacijų taisyklės, kurios atitinka min_confidence ribą.
# itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

# # 3. Rezultatų atvaizdavimas
# print("Dažni elementų deriniai (itemsets):")
# for length, itemset in itemsets.items():
#     print(f"{length}-elementų deriniai:")
#     for items, support in itemset.items():
#         print(f"  Elementai: {items}, support: {support:.2f}")

# # print("\nAsociacijų taisyklės:")
# # Kiekviena taisyklė yra reprezentuojama kaip: X -> Y, su confidence, support ir lift
# # for rule in rules:
# #     print(rule)

# preke = input("Iveskite preke kuria pirkote: ").lower()
# for rule in rules:
#     if rule.lhs == (preke,):
#         print(f"Jei pirksite {preke}, tai greiciausiai pirksite ir {rule.rhs}")
# print(rule.lhs)
# print(rule.rhs)





# for i in range(1,11):
#     print(i)


# print(1)
# print(2)
# print(3)
# print(4)
# print(5)
# print(6)
# print(7)
# print(8)
# print(9)
# print(10)

# import pandas as pd
# pd.read_csv("Online_Retail.csv")
