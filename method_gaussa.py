import numpy as np
from numpy import ndarray
import pandas as pd


def forward_motion(extend_array: ndarray) -> ndarray:
    """
        Прямой ход
        :param extend_array: расширенная матрица
        :return: верхне треугольная расширенная матрица
        """
    length_array = len(extend_array)
    for i in range(length_array):
        extend_array[i] = extend_array[i] / extend_array[i, i]  # ведущий элемент
        for j in range(i + 1, length_array):
            extend_array[j] = extend_array[j] - extend_array[i] * extend_array[j, i]  # опорная строка
    temp_arr = np.round(extend_array, 1)  # округение значений в массиве до одног знака
    return temp_arr


def reverse_motion(_array: ndarray) -> ndarray:
    """
    Обратный ход
    :param _array: верхне треугольная расширенная матрица
    :return: матрица X (решение)
    """
    a_array = _array[:, :3]
    b_array = _array[:, 3]
    reverse_a_array = np.linalg.inv(a_array)  # обратная матрица
    x_arr = np.dot(reverse_a_array, b_array)  # умножение матриц X = A^-1 * b
    return np.array([x_arr])


def print_array(_array: ndarray, col_names: list = None) -> None:
    """
        Вывод матрицы
        :param _array: матрица
        :param col_names: наименование столбцов
        :return: None
        """
    if col_names:
        df = pd.DataFrame(_array, columns=col_names)
        print(df.to_string(index=False))
    else:
        df = pd.DataFrame(_array, columns=col_names)
        print(df.to_string(index=False, header=False))


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
    extend_array: ndarray = np.hstack((array_A, array_b))
    # .copy() нужно чтоб матрица extend_array не изменялась, иначе будет две переменных с ссылкой на один объект
    # (мешает красивому ввыводу данных)
    temp_array = forward_motion(extend_array.copy())
    x_array = reverse_motion(temp_array)
    x_array_col_name = ['x1', 'x2', 'x3']

    print('Матрица A')
    print_array(array_A)
    print('\nМатрица b')
    print_array(array_b)
    print('\nРасширенная матрица')
    print_array(np.round(extend_array, 1))
    print('\nПрямой ход')
    print_array(temp_array)
    print('\nОбратный ход')
    print_array(x_array, x_array_col_name)


if __name__ == '__main__':
    main()
