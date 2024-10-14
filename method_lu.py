import numpy as np
from numpy import ndarray
from method_gaussa import print_array, reverse_motion


def get_L_U_arrays(a_array: ndarray) -> ndarray:
    """
    Получение L и U матриц
    :param a_array: матрица A
    :return: матрицы L и U по формуле (1.11) из учебника
    """
    size_array = a_array.shape[0]
    L = np.array(
        [[0 for _ in range(size_array)] for _ in range(size_array)],
        dtype=np.float64,
    )
    U = np.array(
        [[0 for _ in range(size_array)] for _ in range(size_array)],
        dtype=np.float64,
    )

    for i in range(a_array.shape[0]):
        for j in range(i + 1):
            L[i, j] = a_array[i, j] - sum(L[i, s] * U[s, j] for s in range(j))

        for j in range(i, size_array):
            U[i, j] = ((a_array[i, j] - sum(L[i, s] * U[s, j] for s in range(i))) / L[i, i])

    return L, U


def forward_motion(_array):
    """
    Прямой ход решение Ly=b
    :param _array: расширенная матрица Lb
    :return: матрица y (решение Ly=b)
    """
    a_array = _array[:, :-1]
    b_array = _array[:, -1]
    y = np.array(
        [[0] for _ in range(len(b_array))],
        dtype=np.float64,
    )
    for i in range(len(b_array)):
        y[i] = b_array[i]  # добавляю в матрицу y элемент матрицы B
        for j in range(i):
            y[i] -= a_array[i, j] * y[j]  # тут выражаю элемент x
        y[i] = y[i] / a_array[i, i]  # деление на диагональный элемент
    return y


def main():
    array_A = np.array([
        [2, -1, 0],
        [2, 5, 1],
        [2, 1, -4],
    ], dtype=float)
    array_b = np.array([
        [-5],
        [2],
        [-7],
    ], dtype=float)

    L, U = get_L_U_arrays(array_A)
    extend_Lb_array = np.hstack((L, array_b))
    array_y = forward_motion(extend_Lb_array)
    extend_Uy_array = np.hstack((U, array_y))
    array_x = reverse_motion(extend_Uy_array)  # обратный ход из обратного хода гаусса
    print('Матрица A')
    print_array(array_A)
    print('\n Матрица b')
    print_array(array_b)
    print('\nМатрица L')
    print_array(L)
    print('\nМатрица U')
    print_array(U)
    print('\nРешение Ly = b (матрица Y)')
    print_array(array_y)
    print('\nОтвет: решенеи Ux = y (матрица X)')
    print_array(array_x)


if __name__ == '__main__':
    main()
