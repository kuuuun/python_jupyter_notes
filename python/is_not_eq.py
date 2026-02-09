# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## is, is not, ==
#
# 'is' compares the address

a = 12
b = 12
print(id(a))
print(id(b))
print(a is b)
print(a == b)

list_a=[1,2,3]
list_b=[1,2,3]
print(id(list_a))
print(id(list_b))
print(not list_a is list_b)
print(list_a is not list_b)
print(list_a==list_b)
print(not list_a != list_b)


