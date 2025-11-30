using Random
using LinearAlgebra
using ITensors, ITensorMPS



Random.seed!(1234)


nrow, ncol = 2, 2
D = 2

sites = siteinds("Fermion", nrow*ncol)
# print(sites)