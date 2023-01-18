import csv
import random
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from spisok import *


def methodNumpy(dataset) -> None:
    print("\nNumpy:\n")
    data1: list[int] = []
    data2: list[int] = []
    zarplata: list[int] = []
    for row in dataset:
        data1.append(int(row[3]))
        data2.append(int(row[4]))
        zarplata.append(int(row[7]))
        
    print("Год рождения сотрудников:")    
    print(data1)
    print("Минимальный: ", np.min(data1))
    print("Максимальный: ", np.max(data1))
    print("Средний: ", np.mean(data1, dtype="int"))
    print("Дисперсия: ", np.var(data1))
    print("Стандартное отклонение:", np.std(data1))
    print("Медиана:", np.median(data1))
    print("\n")

    print("Год начала работы в компании:")        
    print(data2)
    print("Минимальный: ", np.min(data2))
    print("Максимальный: ", np.max(data2))
    print("Средний: ", np.mean(data2, dtype="int"))
    print("Дисперсия: ", np.var(data2))
    print("Стандартное отклонение:", np.std(data2))
    print("Медиана:", np.median(data2))
    print("\n")

    print("Зарплата сотрудников:")    
    print(zarplata)
    print("Минимальная: ", np.min(zarplata))
    print("Максимальная: ", np.max(zarplata))
    print("Средняя: ", np.mean(zarplata))
    print("Дисперсия: ", np.var(zarplata))
    print("Стандартное отклонение:", np.std(zarplata))
    print("Медиана:", np.median(zarplata))
    print("\n")

    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(14, 6))

    count_years: list[int] = [0, 0, 0, 0, 0, 0, 0]
    for row in data1:
        if row < 1975:
            count_years[0] = count_years[0] + 1
        if (row >= 1975 and row < 1980):
            count_years[1] = count_years[1] + 1
        if (row >= 1980 and row < 1985):
            count_years[2] = count_years[2] + 1
        if (row >= 1985 and row < 1990):
            count_years[3] = count_years[3] + 1
        if (row >= 1990 and row < 1995):
            count_years[4] = count_years[4] + 1
        if (row >= 1995 and row < 2000):
            count_years[5] = count_years[5] + 1
        if (row >= 2000 and row < 2006):
            count_years[6] = count_years[6] + 1
            
    ax1.step([1975, 1980, 1985, 1990, 1995, 2000, 2006],
             count_years, color = "green")
    ax1.set_title("Год рождения сотрудников")
    
    ax2.hist(data2, color = "orange")
    ax2.set_title("Год начала работы в компании")

    count: list[int] = [0, 0, 0, 0]
    for row in zarplata:
        if row < 100000:
            count[0] = count[0] + 1
        if (row >= 100000 and row < 150000):
            count[1] = count[1] + 1
        if (row >= 150000 and row < 250000):
            count[2] = count[2] + 1
        if (row >= 250000 and row < 300000):
            count[3] = count[3] + 1
            
    title = "20k - 100k", "100k - 150k", "150k - 250k", "250k - 300k"
    explode = (0.1, 0, 0, 0)
    ax3.pie(count, explode, labels = title)
    ax3.set_title("Зарплата сотрудников")
    plt.show()
    
def methodPandas(dataset) -> None:
    print("\nPandas:\n")
    print("Год рождения сотрудников:")
    print("Минимальный: ", dataset["Год рождения"].min())
    print("Максимальный: ", dataset["Год рождения"].max())
    print("Средний: ", dataset["Год рождения"].mean().astype("int64"))
    print("Дисперсия: ", dataset["Год рождения"].var())
    print("Стандартное отклонение:", dataset["Год рождения"].std())
    print("Медиана:", dataset["Год рождения"].median())
    print("\n")

    print("Год начала работы в компании:")        
    print("Минимальный: ", dataset["Год начала работы в компании"].min())
    print("Максимальный: ", dataset["Год начала работы в компании"].max())
    print("Средний: ", dataset["Год начала работы в компании"].mean().astype("int64"))
    print("Дисперсия: ", dataset["Год начала работы в компании"].var())
    print("Стандартное отклонение:", dataset["Год начала работы в компании"].std())
    print("Медиана:", dataset["Год начала работы в компании"].median())
    print("\n")

    print("Зарплата сотрудников:")    
    print("Минимальная: ", dataset["Оклад"].min())
    print("Максимальная: ", dataset["Оклад"].max())
    print("Средняя: ", dataset["Оклад"].mean().astype("int64"))
    print("Дисперсия: ", dataset["Оклад"].var())
    print("Стандартное отклонение:", dataset["Оклад"].std())
    print("Медиана:", dataset["Оклад"].median())
    print("\n")
    

with open("file.csv", mode="w", encoding='utf-16') as file:
    file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
    file_writer.writerow(["Табельный номер", "Фамилия И.О.", "Пол", 
			"Год рождения", "Год начала работы в компании",
			"Подразделение", "Должность", "Оклад",
			"Количество выполненных проектов"])
    
    kol_iter = random.randint(1001, 3000)
    print("Количество строк:")
    print(kol_iter)
    i = 0
    for line in range(1, kol_iter):
        i = i + 1
        tabel_number = i
        
        gend = random.choice(gender)
        if gender == "Мужской":
            name = random.choice(male_last_name)+" "+random.choice(male_name)+" "+random.choice(male_patronymic)
        else:
            name = random.choice(female_last_name)+" "+random.choice(female_name)+" "+random.choice(female_patronymic)

        date_of_birth = random.randint(1970, 2005)
        start_work = date_of_birth + 18
        date_work = random.randint(start_work, 2023)

        depart = random.choice(departments)
        post = random.choice(posts)
        salary = random.randint(20000, 300000)
        projects = (2023 - date_work + 1) * 2
        file_writer.writerow([i, name, gend, date_of_birth, date_work,
                              depart, post, salary, projects])


dataset_DataFrame = pnd.read_csv("file.csv", encoding='utf-16', delimiter=",", lineterminator="\r")
print("Сгенерированный набор данных:")
print(dataset_DataFrame)

with open("file.csv", encoding='utf-16') as dataset_Spisok:
    file_reader = csv.reader(dataset_Spisok)
    for row in file_reader:
        methodNumpy(file_reader)
    
methodPandas(dataset_DataFrame)
        
    
    

    




