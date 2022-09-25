import numpy as np
import matplotlib.pyplot as mpl
 
def funcSumC(matrix, matrix_length):
    summa = 0
    for i in range(matrix_length):
        for j in range(matrix_length):
            if j%2 == 0:
                summa = summa + matrix[i][j]
    return summa


def funcProizvC(matrix, matrix_length):
    proizvedenie = 1
    for i in range(matrix_length):
        if i%2 != 0:
            for j in range(matrix_length):
               proizvedenie  = proizvedenie * matrix[i][j]
    return proizvedenie

 
print("Введите K")
K = int(input())
print("Введите N")
N = int(input())
m = N//2


B = np.random.randint(-10,10,(m,m))
print("Подматрица B:")
print(B,'\n')

C = np.random.randint(-10,10,(m,m))
print("Подматрица C:")
print(C,'\n')

D = np.random.randint(-10,10,(m,m))
print("Подматрица D:")
print(D,'\n')

E = np.random.randint(-10,10,(m,m))
print("Подматрица E:")
print(E,'\n\n')

A = np.vstack((np.hstack((B,C)), np.hstack((D,E))))
print("Матрица A:");
print(A,'\n')


F = A.copy()

summ = funcSumC(C, m)
print("Сумма в нечетных столбцах подматрицы С:")
print(summ)

proiz = funcProizvC(C, m)
print("Произведение в четных строках подматрицы С:")
print(proiz, '\n')

if summ > proiz:
    B1 = np.flip(B, axis = 1)
    C1 = np.flip(C, axis = 1)
    F = np.vstack([np.hstack([C1, B1]), np.hstack([D, E])])
else:
    F = np.hstack([np.vstack([E, B]), np.vstack([D, C])]) 
print("Матрица F:")
print(F,'\n')


A1 = np.linalg.inv(A)
F1 = np.linalg.inv(F)
G = np.tril(A)
G1 = np.linalg.inv(G)
if np.linalg.det(A) > np.diag(F).sum():
    vyrazhenie = A1 * A.T - K * F1
else:
    vyrazhenie = (A.T + G1 + F1) * K
print("Найденное выражение:")
print(vyrazhenie, '\n')


mpl.subplot(2,2,1)
mpl.plot(F[:m, :m])
mpl.subplot(2,2,2)
mpl.plot(F[:m, m:])
mpl.subplot(2,2,3)
mpl.plot(F[m:, :m])
mpl.subplot(2,2,4)
mpl.plot(F[m:, m:])
mpl.show()

mpl.subplot(2, 2, 1)
mpl.imshow(F[:m, :m], cmap='rainbow', interpolation='bilinear')
mpl.subplot(2, 2, 2)
mpl.imshow(F[:m, m:], cmap='rainbow', interpolation='bilinear')
mpl.subplot(2, 2, 3)
mpl.imshow(F[m:, :m], cmap='rainbow', interpolation='bilinear')
mpl.subplot(2, 2, 4)
mpl.imshow(F[m:, m:], cmap='rainbow', interpolation='bilinear')
mpl.show()



