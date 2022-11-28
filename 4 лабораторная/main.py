import csv
import numpy as np
import pandas as pnd
import pylab
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def products(path: Path) -> None:
    with open(path, 'w', encoding='utf-16', newline='') as file:
            titles = [
                'Продукт',
                'Сладость',
                'Хруст',
                'Класс'
             ]
            file_writer = csv.DictWriter(file, fieldnames = titles)
            file_writer.writeheader()

            products_arrays = [['Яблоко', 7, 7, 0],
                               ['Салат', 2, 5, 1],
                               ['Бекон', 1, 2, 2],
                               ['Банан', 9, 1, 0],
                               ['Орехи', 1, 5, 2],
                               ['Рыба', 1, 1, 2],
                               ['Сыр', 1, 1, 2],
                               ['Виноград', 8, 1, 0],
                               ['Морковь', 2, 8, 1],
                               ['Апельсин', 6, 1, 0]]

            for i in range(0, len(products_arrays)):
                product_name: str = products_arrays[i][0]
                sweet: int = products_arrays[i][1]
                crunch: int = products_arrays[i][2]
                class_name: str = products_arrays[i][3]

                file_writer.writerow({
                    'Продукт': product_name,
                    'Сладость': sweet,
                    'Хруст': crunch,
                    'Класс': class_name
                })


def euclid_distance(x1, y1, x2, y2):
    distance = (abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2) ** 0.5
    return distance


def knn_method(products_dataset, teach, pars_window):
    products_data = np.array(products_dataset)
    test = len(products_data) - teach
    count_objects = 0
    classes = [0] * test
    euclide_matrix = np.zeros((test, teach))

    for i in range(test):
        for j in range(teach):
            distance_value = euclid_distance(int(products_data[teach + i][1]), int(products_data[teach + i][2]), int(products_data[j + 1][1]), int(products_data[j + 1][2]))
            euclide_matrix[i][j] = distance_value if distance_value < pars_window else 1000

    for i in range(test):
        print(str(i) + '. Классификация ', products_data[teach + i][0])
        weights = [0] * products_dataset.iloc[:]['Класс'].nunique()
        neighbor = np.sum(euclide_matrix[i] != 1000)
        
        for j in range(neighbor + 1):
            ind_min = euclide_matrix[i].argmin()
            weights[int(products_data[ind_min + 1][3])] += ((neighbor - j + 1) / neighbor)
            euclide_matrix[i][ind_min] = 1000
            print('Индекс соседа =', ind_min, 'Сосед -', products_data[ind_min + 1][0])
            
        classes[i] = np.array(weights).argmax()
        print('Полученный элемент =', classes[i], 'Реальный класс элемента =', products_data[teach + i][3])
         
        if int(classes[i]) != int(products_data[teach + i][3]):
            print('Не совпал')
            
        else:
            print('Совпал')
            count_objects += 1
            
    print(classes)
    print('Количество совпадений:', str(count_objects))
    return classes
    

def knn_sklearn(products_dataset) -> None:
    x = products_dataset.iloc[:, 1:3].values
    y = products_dataset.iloc[:, 3].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=False, stratify=None)

    classifier = KNeighborsClassifier(n_neighbors=4)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))


def vizualization(products_dataset, window) -> None:
    colours_products = {'0': 'blue', '1': 'red', '2': 'yellow', '3': 'green'}
    sweet = products_dataset['Сладость']
    crunch = products_dataset['Хруст']
    col_list = [colours_products[str(i)] for i in products_dataset['Класс']]
    pylab.subplot(2, 1, window)
    plt.scatter(sweet, crunch, c=col_list)
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')


def products_new(path: Path) -> None:
    with open(path, 'w', encoding='utf-16', newline='') as file:
            titles = [
                'Продукт',
                'Сладость',
                'Хруст',
                'Класс'
             ]
            file_writer = csv.DictWriter(file, fieldnames = titles)
            file_writer.writeheader()

            products_arrays = [['Яблоко', 7, 7, 0],
                               ['Салат', 2, 5, 1],
                               ['Смородина', 3, 1, 3],
                               ['Малина', 4, 1, 3],
                               ['Бекон', 1, 2, 2],
                               ['Банан', 9, 1, 0],
                               ['Клубника', 8, 1, 3],
                               ['Орехи', 1, 5, 2],
                               ['Крыжовник', 3, 1, 3],
                               ['Рыба', 1, 1, 2],
                               ['Сыр', 1, 1, 2],
                               ['Виноград', 8, 1, 0],
                               ['Морковь', 2, 8, 1],
                               ['Арбуз', 9, 7, 3],
                               ['Апельсин', 6, 1, 0]]

            for i in range(0, len(products_arrays)):
                product_name: str = products_arrays[i][0]
                sweet: int = products_arrays[i][1]
                crunch: int = products_arrays[i][2]
                class_name: str = products_arrays[i][3]

                file_writer.writerow({
                    'Продукт': product_name,
                    'Сладость': sweet,
                    'Хруст': crunch,
                    'Класс': class_name
                })


file_path = Path(Path.cwd(), 'file.csv')
file_path_new = Path(Path.cwd(), 'file_new.csv')
products(file_path)
products_new(file_path_new)

products_dataset_1 = pnd.read_csv(file_path, encoding='utf-16')
start_data = products_dataset_1[:len(products_dataset_1)]['Класс']
set_1 = pnd.Series(knn_method(products_dataset_1, 9, 4), dtype=pnd.StringDtype())
start_data = pnd.concat([start_data, set_1])
colours = {'0': 'red', '1': 'green', '2': 'blue'}
vizualization(products_dataset_1, 1)
colour_list = [colours[str(i)] for i in start_data]
vizualization(products_dataset_1, 2)
plt.show()

products_dataset_2 = pnd.read_csv(file_path_new, encoding='utf-16')
set_1 = pnd.Series(knn_method(products_dataset_2, 15, 4), dtype=pnd.StringDtype())    
start_data = pnd.concat([start_data, set_1])
vizualization(products_dataset_2, 1)
vizualization(products_dataset_2, 2)                   
plt.show()

knn_sklearn(products_dataset_1)
knn_sklearn(products_dataset_2)
