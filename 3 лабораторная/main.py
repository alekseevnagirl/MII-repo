import csv
import numpy as np
import pandas as pnd
import random as rand
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from spisok import *

def people(path: Path) -> None:
    summa_strock: int = rand.randint(1000, 2000)
    
    with open(path, 'w', encoding='utf-16', newline='') as file:
        titles = [
            'Табельный номер',
            'Фамилия И.О.',
            'Пол',
            'Год рождения',
            'Год начала работы в компании',
            'Подразделение',
            'Должность',
            'Оклад',
            'Количество выполненных проектов'
        ]
        
        file_writer = csv.DictWriter(file, fieldnames=titles)
        file_writer.writeheader()
        
        for i in range(1, summa_strock + 1):
            choice_gender: str = rand.choice(gender)
            
            if choice_gender == 'Женский':
                full_name = rand.choice(female_last_name) + ' ' + rand.choice(female_name) + ' ' + rand.choice(female_patronymic)
            else:
                full_name = rand.choice(male_last_name) + ' ' + rand.choice(male_name) + ' ' + rand.choice(male_patronymic)
                
            year_of_birth: int = rand.randint(1960, 2004)
            year_of_start_working: int = year_of_birth + rand.randint(18, 30)
            department: str = rand.choice(departments)
            post: str = rand.choice(posts)
            salary: int = rand.randint(20000, 200000)
            kol_projects: int = rand.randint(1, 10)
            
            file_writer.writerow({
                'Табельный номер': i,
                'Фамилия И.О.': full_name,
                'Пол': choice_gender,
                'Год рождения': year_of_birth,
                'Год начала работы в компании': year_of_start_working,
                'Подразделение': department,
                'Должность': post,
                'Оклад': salary,
                'Количество выполненных проектов': kol_projects
            })

def statistics_lists(path: Path) -> None:
    with open(path, 'r', encoding='utf-16') as file:
        kol_projects: list[int] = []
        salaries: list[int] = []
        years_of_birth: list[int] = []
        years_of_start_working: list[int] = []
        departments: list[str] = []

        file_reader = csv.DictReader(file)
        for line in file_reader:
            kol_projects.append(int(line['Количество выполненных проектов']))
            salaries.append(int(line['Оклад']))
            years_of_birth.append(int(line['Год рождения']))
            years_of_start_working.append(int(line['Год начала работы в компании']))
            departments.append(line['Подразделение'])

        work_experiencies: list[int] = [datetime.now().year - years_of_start_working[i] for i in range(len(years_of_birth))]

        print('Numpy')
        print('')
        print(f'Количество сотрудников: {np.count_nonzero(work_experiencies)}')
        print('')
        print('')
        
        print('Опыт работы')
        print('')
        print(f'Минимальный стаж: {np.min(work_experiencies)}')
        print(f'Максимальный стаж: {np.max(work_experiencies)}')
        print(f'Средний стаж: {round(np.average(work_experiencies), 2)}')
        print('')
        print('')

        print('Заработная плата')
        print('')
        print(f'Минимальная зарплата, в рублях: {np.min(salaries)}')
        print(f'Максимальная зарплата, в рублях: {np.max(salaries)}')
        print(f'Средняя зарплата, в рублях: {round(np.average(salaries), 2)}')
        print(f'Медианное значение зарплаты, в рублях: {np.median(salaries)}')
        print(f'Дисперсия зарплаты, в рублях: {round(np.var(salaries), 2)}')
        print(f'Стандартное отклонение зарплаты, в рублях: {round(np.std(salaries), 2)}')
        print('')
        print('')
        
        print('Проекты')
        print('')
        print(f'Максимальное количество проектов на одного сотрудника: {np.max(kol_projects)}')
        print(f'Минимальное количество проектов на одного сотрудника: {np.min(kol_projects)}')
        print(f'Среднее количество проектов на одного сотрудника: {round(np.average(kol_projects), 2)}')
        print(f'Количество выполненных проектов: {np.sum(kol_projects)}')
        print('')
        print('')

def statistics_pnd(path: Path) -> None:
    company = pnd.read_csv(path, encoding='utf-16')
    
    print('Pandas')
    print('')
    print(f'Количество сотрудников: {company["Табельный номер"].count()}')
    print('')
    print('')
    
    print('Проекты')
    print('')
    print(f'Максимальное количество проектов на одного сотрудника: {company["Количество выполненных проектов"].max()}')
    print(f'Минимальное количество проектов на одного сотрудника: {company["Количество выполненных проектов"].min()}')
    print(f'Среднее количество проектов на одного сотрудника: {round(company["Количество выполненных проектов"].sum() / company["Табельный номер"].count(), 2)}')
    print(f'Количество выполненных проектов: {company["Количество выполненных проектов"].sum()}')
    print('')
    print('')
    
    print('Зарплата')
    print('')
    print(f'Максимальная зарплата, в рублях: {company["Оклад"].max()}')
    print(f'Минимальная зарплата, в рублях: {company["Оклад"].min()}')
    print(f'Средняя зарплата, в рублях: {round(company["Оклад"].sum() / company["Оклад"].count(), 2)}')
    print(f'Медианное значение зарплаты, в рублях: {company["Оклад"].median()}')
    print(f'Дисперсия зарплаты, в рублях: {round(company["Оклад"].var(), 2)}')
    print(f'Стандартное отклонение зарплаты, в рублях: {round(company["Оклад"].std(), 2)}')
    print('')
    print('')
    
    print('Отделы')
    print('')
    print(f'Количество отделов: {len(np.unique(departments))}')
    print(f'Количество сотрудников в отделе Java разработки: {len(company[company["Подразделение"] == "Отдел Java разработки"])}')
    print(f'Количество сотрудников в отделе Python разработки: {len(company[company["Подразделение"] == "Отдел Python разработки"])}')
    print(f'Количество сотрудников в отделе Frontend разработки: {len(company[company["Подразделение"] == "Отдел frontend разработки"])}')
    print(f'Количество сотрудников в отделе PHP разработки: {len(company[company["Подразделение"] == "Отдел PHP разработки"])}')
    print('')
    print('')

def graphic_statistics(path: Path) -> None:
    company = pnd.read_csv(path, encoding='utf-16')
    projects_statistics = {}
    jobs = company['Должность'].unique()

    for item in jobs:
        subdiv_employees = company[company['Должность'] == item]
        projects_statistics[item] = round(subdiv_employees['Всего проектов'].sum() / subdiv_employees['Всего проектов'].count(), 2)

    plt.plot(company["Оклад"], label='Оклад')
    plt.axhline(y=np.nanmean(company['Оклад'].mean()), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.title('Динамика зарплаты и среднее значение зарплаты', loc='center')
    plt.show()

    plt.bar(projects_statistics.keys(), projects_statistics.values())
    plt.title('Количество выполненных проектов по должностям', loc='center')
    plt.show()

    plt.hist(company['Подразделение'], bins=15)
    plt.title('Количесство сотрудников по отделам', loc='center')
    plt.show()


file_path = Path(Path.cwd(), 'file.csv')
people(file_path)
statistics_lists(file_path)
statistics_pnd(file_path)
graphic_statistics(file_path)
