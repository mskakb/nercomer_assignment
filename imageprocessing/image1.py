#行列の四則演算
import numpy as np

mat1 = np.array([[1,3,5],[6,4,9],[1,9,2]])
mat2 = np.array([[2,9,8],[8,2,5],[4,7,8]])
mat3 = np.empty((3,3))

#加算
mat3 = mat1+mat2
print(mat3)

#減算
print(mat1-mat2)

#乗算
print(4*mat1)

#行列同士の積
print(mat1@mat2)

#逆行列
mat1_inv = np.linalg.inv(mat1)
print(mat1_inv)